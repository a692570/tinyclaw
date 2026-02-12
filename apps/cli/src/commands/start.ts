/**
 * Start Command
 *
 * Boots the TinyClaw agent: initializes all subsystems, verifies provider
 * connectivity, and starts the Web UI / API server.
 *
 * Pre-flight check: ensures at least one provider API key is configured
 * via secrets-engine before proceeding. If not, directs the user to run
 * `tinyclaw setup`.
 */

import { join } from 'path';
import { homedir } from 'os';
import {
  createDatabase,
  agentLoop,
  createOllamaProvider,
  ProviderOrchestrator,
  logger,
  HeartwareManager,
  createHeartwareTools,
  loadHeartwareContext,
  createLearningEngine,
  SecretsManager,
  ConfigManager,
  createSecretsTools,
  createConfigTools,
  createSessionQueue,
  createCronScheduler,
  loadPlugins,
  buildProviderKeyName,
  type HeartwareConfig,
  type ChannelPlugin,
  type ProviderTierConfig,
  type Provider,
} from '@tinyclaw/core';
import { createWebUI } from '@tinyclaw/ui';
import { theme } from '../ui/theme.js';

/**
 * Run the agent start flow
 */
export async function startCommand(): Promise<void> {
  logger.log('🐜 TinyClaw — Small agent, mighty friend');

  // --- Data directory ---------------------------------------------------

  const dataDir = process.env.TINYCLAW_DATA_DIR || join(homedir(), '.tinyclaw');
  logger.info('📂 Data directory:', { dataDir });

  // --- Initialize secrets engine ----------------------------------------

  let secretsManager: SecretsManager;

  try {
    secretsManager = await SecretsManager.create();
  } catch (err: unknown) {
    // Detect IntegrityError from @wgtechlabs/secrets-engine
    // The HMAC stored in meta.json does not match the database contents.
    // This may indicate file corruption, tampering, or a partial write.
    if (
      err instanceof Error &&
      'code' in err &&
      (err as { code: string }).code === 'INTEGRITY_ERROR'
    ) {
      const storePath = join(homedir(), '.secrets-engine');

      console.log();
      console.log(theme.error('  ✖ Secrets store integrity check failed.'));
      console.log();
      console.log('    The secrets store may have been corrupted or tampered with.');
      console.log('    This can happen due to disk errors, power loss, or external changes.');
      console.log();
      console.log('    To resolve, delete the store and re-run setup:');
      console.log();
      console.log(`      1. ${theme.cmd(`rm -rf ${storePath}`)}`);
      console.log(`      2. ${theme.cmd('tinyclaw setup')}`);
      console.log();
      process.exit(1);
    }

    throw err;
  }

  logger.info('✅ Secrets engine initialized', {
    storagePath: secretsManager.storagePath,
  });

  // --- Pre-flight: check for provider API key --------------------------

  const hasOllamaKey = await secretsManager.check(
    buildProviderKeyName('ollama')
  );

  if (!hasOllamaKey) {
    console.log();
    console.log(
      theme.error('  ✖ No provider API key found.')
    );
    console.log(
      `    Run ${theme.cmd('tinyclaw setup')} to configure your provider.`
    );
    console.log();
    await secretsManager.close();
    process.exit(1);
  }

  // --- Initialize config engine -----------------------------------------

  const configManager = await ConfigManager.create();
  logger.info('✅ Config engine initialized', { configPath: configManager.path });

  // Read provider settings from config (fallback to defaults)
  const providerModel =
    configManager.get<string>('providers.starterBrain.model') ?? 'gpt-oss:120b-cloud';
  const providerBaseUrl =
    configManager.get<string>('providers.starterBrain.baseUrl') ?? 'https://ollama.com';

  // --- Initialize database ----------------------------------------------

  const dbPath = join(dataDir, 'data', 'agent.db');
  const db = createDatabase(dbPath);
  logger.info('✅ Database initialized');

  // --- Initialize learning engine ---------------------------------------

  const learningPath = join(dataDir, 'learning');
  const learning = createLearningEngine({ storagePath: learningPath });
  logger.info('✅ Learning engine initialized');

  // --- Initialize heartware ---------------------------------------------

  const heartwareConfig: HeartwareConfig = {
    baseDir: join(dataDir, 'heartware'),
    userId: 'default-user',
    auditDir: join(dataDir, 'audit'),
    backupDir: join(dataDir, 'heartware', '.backups'),
    maxFileSize: 1_048_576, // 1 MB
  };

  const heartwareManager = new HeartwareManager(heartwareConfig);
  await heartwareManager.initialize();
  logger.info('✅ Heartware initialized');

  const heartwareContext = await loadHeartwareContext(heartwareManager);
  logger.info('✅ Heartware context loaded');

  // --- Initialize default provider (reads key from secrets-engine) ------

  const defaultProvider = createOllamaProvider({
    secrets: secretsManager,
    model: providerModel,
    baseUrl: providerBaseUrl,
  });

  // Verify default provider is reachable
  await new ProviderOrchestrator({ defaultProvider }).selectActiveProvider();
  logger.info('✅ Default provider initialized and verified');

  // --- Load plugins ------------------------------------------------------

  const plugins = await loadPlugins(configManager);
  logger.info('✅ Plugins loaded', {
    channels: plugins.channels.length,
    providers: plugins.providers.length,
    tools: plugins.tools.length,
  });

  // --- Initialize plugin providers ---------------------------------------

  const pluginProviders: Provider[] = [];

  for (const pp of plugins.providers) {
    try {
      const provider = await pp.createProvider(secretsManager);
      pluginProviders.push(provider);
      logger.info(`✅ Plugin provider initialized: ${pp.name} (${provider.id})`);
    } catch (err) {
      logger.error(`Failed to initialize provider plugin "${pp.name}":`, err);
    }
  }

  // --- Initialize smart routing orchestrator -----------------------------

  const tierMapping = configManager.get<ProviderTierConfig>('routing.tierMapping');

  const orchestrator = new ProviderOrchestrator({
    defaultProvider,
    providers: pluginProviders,
    tierMapping: tierMapping ?? undefined,
  });

  logger.info('✅ Smart routing initialized', {
    providers: orchestrator.getRegistry().ids(),
    tierMapping: tierMapping ?? 'all-default',
  });

  // --- Initialize tools -------------------------------------------------

  const tools = [
    ...createHeartwareTools(heartwareManager),
    ...createSecretsTools(secretsManager),
    ...createConfigTools(configManager),
  ];

  // Merge plugin pairing tools (channels + providers)
  const pairingTools = [
    ...plugins.channels.flatMap(
      (ch) => ch.getPairingTools?.(secretsManager, configManager) ?? [],
    ),
    ...plugins.providers.flatMap(
      (pp) => pp.getPairingTools?.(secretsManager, configManager) ?? [],
    ),
  ];

  // Create a temporary context for plugin tools that need AgentContext
  const baseContext = {
    db,
    provider: defaultProvider,
    learning,
    tools,
    heartwareContext,
    secrets: secretsManager,
    configManager,
  };

  const pluginTools = plugins.tools.flatMap(
    (tp) => tp.createTools(baseContext),
  );

  const allTools = [...tools, ...pairingTools, ...pluginTools];
  logger.info('✅ Loaded tools', { count: allTools.length });

  // --- Create agent context ---------------------------------------------

  const context = {
    db,
    provider: defaultProvider,
    learning,
    tools: allTools,
    heartwareContext,
    secrets: secretsManager,
    configManager,
  };

  // --- Initialize session queue ------------------------------------------

  const queue = createSessionQueue();
  logger.info('✅ Session queue initialized');

  // --- Initialize cron scheduler -----------------------------------------

  const cron = createCronScheduler();

  cron.register({
    id: 'memory-consolidation',
    schedule: '24h',
    handler: async () => {
      await queue.enqueue('heartbeat', async () => {
        await agentLoop(
          'Review your recent memory logs and consolidate any important patterns or facts into long-term memory. Be brief.',
          'heartbeat',
          context,
        );
      });
    },
  });

  cron.start();
  logger.info('✅ Cron scheduler initialized');

  // --- Start Web UI / API server ----------------------------------------

  const port = parseInt(process.env.PORT || '3000', 10);
  const webUI = createWebUI({
    port,
    onMessage: async (message: string, userId: string) => {
      const { provider, classification, failedOver } =
        await orchestrator.routeWithHealth(message);
      logger.debug('Routed query', {
        tier: classification.tier,
        provider: provider.id,
        confidence: classification.confidence.toFixed(2),
        failedOver,
      });
      const routedContext = { ...context, provider };
      return await queue.enqueue(userId, () =>
        agentLoop(message, userId, routedContext),
      );
    },
    onMessageStream: async (message: string, userId: string, callback) => {
      const { provider, classification, failedOver } =
        await orchestrator.routeWithHealth(message);
      logger.debug('Routed query (stream)', {
        tier: classification.tier,
        provider: provider.id,
        failedOver,
      });
      const routedContext = { ...context, provider };
      await queue.enqueue(userId, () =>
        agentLoop(message, userId, routedContext, callback),
      );
    },
  });

  await webUI.start();

  // --- Start channel plugins ---------------------------------------------

  const pluginRuntimeContext = {
    enqueue: async (userId: string, message: string) => {
      const { provider } = await orchestrator.routeWithHealth(message);
      const routedContext = { ...context, provider };
      return queue.enqueue(userId, () =>
        agentLoop(message, userId, routedContext),
      );
    },
    agentContext: context,
    secrets: secretsManager,
    configManager,
  };

  for (const channel of plugins.channels) {
    try {
      await channel.start(pluginRuntimeContext);
      logger.info(`✅ Channel plugin started: ${channel.name}`);
    } catch (err) {
      logger.error(`Failed to start channel plugin "${channel.name}":`, err);
    }
  }

  const stats = learning.getStats();
  logger.log(`🧠 Learning: ${stats.totalPatterns} patterns learned`);
  logger.log('');
  logger.log('🎉 TinyClaw is ready!');
  logger.log(`   API server: http://localhost:${port}`);
  logger.log('   Web UI: Run "bun run dev:ui" then open http://localhost:5173');
  logger.log('');

  // --- Graceful shutdown ------------------------------------------------

  let isShuttingDown = false;

  process.on('SIGINT', async () => {
    if (isShuttingDown) {
      logger.info('Shutdown already in progress, ignoring signal');
      return;
    }
    isShuttingDown = true;
    logger.info('👋 Shutting down TinyClaw...');

    // 0. Cron scheduler + session queue
    try {
      cron.stop();
      queue.stop();
      logger.info('Cron scheduler and session queue stopped');
    } catch (err) {
      logger.error('Error stopping cron/queue:', err);
    }

    // 0.5. Channel plugins
    for (const channel of plugins.channels) {
      try {
        await channel.stop();
        logger.info(`Channel plugin stopped: ${channel.name}`);
      } catch (err) {
        logger.error(`Error stopping channel plugin "${channel.name}":`, err);
      }
    }

    // 1. Web UI
    try {
      if (typeof (webUI as any).stop === 'function') {
        await (webUI as any).stop();
      } else if (typeof (webUI as any).close === 'function') {
        await (webUI as any).close();
      }
      logger.info('Web UI stopped');
    } catch (err) {
      logger.error('Error stopping Web UI:', err);
    }

    // 2. Learning engine
    try {
      if (typeof (learning as any).close === 'function') {
        await (learning as any).close();
      }
      logger.info('Learning engine closed');
    } catch (err) {
      logger.error('Error closing learning engine:', err);
    }

    // 3. Heartware
    try {
      if (typeof (heartwareManager as any).close === 'function') {
        await (heartwareManager as any).close();
      }
      logger.info('Heartware manager closed');
    } catch (err) {
      logger.error('Error closing heartware manager:', err);
    }

    // 4. Config engine
    try {
      configManager.close();
      logger.info('Config engine closed');
    } catch (err) {
      logger.error('Error closing config engine:', err);
    }

    // 5. Secrets engine
    try {
      await secretsManager.close();
      logger.info('Secrets engine closed');
    } catch (err) {
      logger.error('Error closing secrets engine:', err);
    }

    // 6. Database (last — other services may flush here)
    try {
      db.close();
      logger.info('Database closed');
    } catch (err) {
      logger.error('Error closing database:', err);
    }

    process.exit(0);
  });
}
