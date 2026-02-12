/**
 * Dynamic Agent Delegation
 *
 * Provides ephemeral sub-agent spawning for the primary agent. The primary
 * agent can delegate focused tasks (research, analysis, summarization) to
 * a sub-agent via the `delegate_task` tool. Sub-agents run a lightweight
 * agent loop in-memory — no database writes, no learning, no persistence.
 *
 * Sub-agents:
 *   - Are created with a role and task description
 *   - Get a filtered set of tools (default: read-only)
 *   - Can be routed to a specific provider tier
 *   - Cannot spawn their own sub-agents (no recursion)
 *   - Are destroyed after returning a result
 */

import { logger } from './logger.js';
import type { Provider, Tool, Message, ToolCall } from './types.js';
import type { ProviderOrchestrator } from './router/orchestrator.js';
import type { QueryTier } from './router/classifier.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SUB_AGENT_MAX_ITERATIONS = 10;
const SUB_AGENT_TIMEOUT_MS = 60_000;
const TOOL_ACTION_KEYS = ['action', 'tool', 'name'];

/** Tools safe for sub-agents by default (read-only). */
const DEFAULT_SAFE_TOOLS = [
  'heartware_read',
  'heartware_search',
  'heartware_list',
  'memory_recall',
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SubAgentConfig {
  /** The task description to accomplish. */
  task: string;
  /** Role description for the sub-agent's system prompt. */
  role: string;
  /** Provider to use for LLM calls. */
  provider: Provider;
  /** Tools available to the sub-agent. */
  tools: Tool[];
  /** Max execution time in ms (default: 60_000). */
  timeout?: number;
}

export interface SubAgentResult {
  /** Whether the sub-agent completed successfully. */
  success: boolean;
  /** The final text response from the sub-agent. */
  response: string;
  /** Number of iterations the sub-agent took. */
  iterations: number;
  /** Provider that was used. */
  providerId: string;
}

export interface DelegationToolConfig {
  /** Provider orchestrator for tier-based routing. */
  orchestrator: ProviderOrchestrator;
  /** All available tools (delegation will filter from this set). */
  allTools: Tool[];
  /** Override default safe tool names for sub-agents. */
  defaultSubAgentTools?: string[];
}

// ---------------------------------------------------------------------------
// Tool call parsing helpers
// (Duplicated from agent.ts — pure functions, candidates for future
//  extraction into core/src/tool-utils.ts)
// ---------------------------------------------------------------------------

function normalizeToolArguments(
  args: Record<string, unknown>,
): Record<string, unknown> {
  const normalized = { ...args };

  if (!('filename' in normalized) && 'file_path' in normalized) {
    normalized.filename = normalized.file_path;
  }

  if (!('filename' in normalized) && 'path' in normalized) {
    normalized.filename = normalized.path;
  }

  return normalized;
}

function extractToolCallFromText(text: string): ToolCall | null {
  if (!text) return null;

  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) return null;

  const raw = text.slice(start, end + 1).trim();
  let parsed: Record<string, unknown>;

  try {
    parsed = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }

  if (!parsed || typeof parsed !== 'object') return null;

  const actionKey = TOOL_ACTION_KEYS.find((key) => key in parsed);
  const toolName = actionKey ? String(parsed[actionKey]) : '';
  if (!toolName) return null;

  const { action, tool, name, ...rest } = parsed as Record<string, unknown>;

  return {
    id: crypto.randomUUID(),
    name: toolName,
    arguments: normalizeToolArguments(rest),
  };
}

// ---------------------------------------------------------------------------
// Sub-Agent Runner
// ---------------------------------------------------------------------------

/**
 * Run an ephemeral sub-agent to completion.
 *
 * The sub-agent runs a lightweight agent loop entirely in-memory:
 * - No database writes
 * - No learning engine analysis
 * - No conversation history or compaction
 *
 * Returns a result object — never throws.
 */
export async function runSubAgent(
  config: SubAgentConfig,
): Promise<SubAgentResult> {
  const { task, role, provider, tools } = config;
  const timeout = config.timeout ?? SUB_AGENT_TIMEOUT_MS;

  const systemPrompt =
    `You are a focused sub-agent with the role: ${role}.\n\n` +
    `Complete the following task and return a clear, concise result.\n` +
    `Do not ask follow-up questions — use your best judgment and available tools.\n` +
    `When you have finished, respond with your final answer as plain text.`;

  const messages: Message[] = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: task },
  ];

  let iterations = 0;

  const run = async (): Promise<SubAgentResult> => {
    for (let i = 0; i < SUB_AGENT_MAX_ITERATIONS; i++) {
      iterations = i + 1;

      const response = await provider.chat(messages, tools);

      // --- Text response ---------------------------------------------------
      if (response.type === 'text') {
        const toolCall = extractToolCallFromText(response.content || '');

        if (toolCall) {
          // JSON-in-text tool call — execute and continue
          const result = await executeToolCall(toolCall, tools);
          messages.push({
            role: 'assistant',
            content: response.content || '',
          });
          messages.push({
            role: 'tool',
            content: result,
            toolCallId: toolCall.id,
          });
          continue;
        }

        // Pure text — this is the final response
        return {
          success: true,
          response: response.content || '',
          iterations,
          providerId: provider.id,
        };
      }

      // --- Native tool calls ------------------------------------------------
      if (response.type === 'tool_calls' && response.toolCalls?.length) {
        const assistantContent = response.content ?? '';

        for (const tc of response.toolCalls) {
          const result = await executeToolCall(tc, tools);
          messages.push({
            role: 'assistant',
            content: assistantContent,
            toolCalls: [tc],
          });
          messages.push({
            role: 'tool',
            content: result,
            toolCallId: tc.id,
          });
        }
        continue;
      }
    }

    return {
      success: false,
      response: 'Sub-agent reached maximum iterations without completing the task.',
      iterations,
      providerId: provider.id,
    };
  };

  try {
    // Wrap in timeout
    const result = await Promise.race([
      run(),
      new Promise<SubAgentResult>((_, reject) =>
        setTimeout(() => reject(new Error('Sub-agent timed out')), timeout),
      ),
    ]);
    return result;
  } catch (err) {
    logger.error('Sub-agent error:', err);
    return {
      success: false,
      response: `Sub-agent error: ${(err as Error).message}`,
      iterations,
      providerId: provider.id,
    };
  }
}

// ---------------------------------------------------------------------------
// Tool execution helper
// ---------------------------------------------------------------------------

async function executeToolCall(
  toolCall: ToolCall,
  tools: Tool[],
): Promise<string> {
  const tool = tools.find((t) => t.name === toolCall.name);
  if (!tool) {
    return `Error: Tool "${toolCall.name}" not found`;
  }

  try {
    return await tool.execute(toolCall.arguments);
  } catch (err) {
    return `Error: ${(err as Error).message}`;
  }
}

// ---------------------------------------------------------------------------
// Delegation Tool Factory
// ---------------------------------------------------------------------------

/**
 * Create the `delegate_task` tool that the primary agent can use to spawn
 * ephemeral sub-agents.
 *
 * The tool captures `orchestrator` and `allTools` via closure — same pattern
 * as `createSecretsTools()` and `createConfigTools()`.
 */
export function createDelegationTool(config: DelegationToolConfig): Tool {
  const { orchestrator, allTools } = config;
  const safeToolNames = config.defaultSubAgentTools ?? DEFAULT_SAFE_TOOLS;

  return {
    name: 'delegate_task',
    description:
      'Delegate a task to an ephemeral sub-agent. The sub-agent runs independently with ' +
      'a specific role, completes the task, and returns the result. Use this for research, ' +
      'analysis, data gathering, summarization, or any task that benefits from focused ' +
      'single-purpose execution. The sub-agent has read-only access to heartware and ' +
      'memory by default. You can optionally grant additional tools and route to a ' +
      'specific provider tier (e.g., "reasoning" for complex analysis).',
    parameters: {
      type: 'object',
      properties: {
        task: {
          type: 'string',
          description:
            'Clear, detailed description of what the sub-agent should accomplish',
        },
        role: {
          type: 'string',
          description:
            'Role/specialty for the sub-agent (e.g., "Research Specialist", ' +
            '"Data Analyst", "Content Summarizer")',
        },
        tier: {
          type: 'string',
          description:
            'Optional complexity tier for provider routing. If omitted, ' +
            'the task text is auto-classified.',
          enum: ['simple', 'moderate', 'complex', 'reasoning'],
        },
        tools: {
          type: 'array',
          items: { type: 'string' },
          description:
            'Optional additional tool names to grant the sub-agent beyond the ' +
            'default read-only set (heartware_read, heartware_search, heartware_list, memory_recall)',
        },
      },
      required: ['task', 'role'],
    },

    async execute(args: Record<string, unknown>): Promise<string> {
      const task = args.task as string;
      const role = args.role as string;
      const tierOverride = args.tier as string | undefined;
      const additionalToolNames = (args.tools as string[]) || [];

      if (!task?.trim()) {
        return 'Error: task must be a non-empty string.';
      }
      if (!role?.trim()) {
        return 'Error: role must be a non-empty string.';
      }

      // 1. Resolve provider
      let provider;
      try {
        if (tierOverride) {
          provider = orchestrator
            .getRegistry()
            .getForTier(tierOverride as QueryTier);
        } else {
          const routeResult = await orchestrator.routeWithHealth(task);
          provider = routeResult.provider;
        }
      } catch (err) {
        return `Error resolving provider: ${(err as Error).message}`;
      }

      logger.info('Delegating task', {
        role,
        tier: tierOverride ?? 'auto',
        provider: provider.id,
        additionalTools: additionalToolNames,
      });

      // 2. Assemble tool set
      const allowedToolNames = new Set([
        ...safeToolNames,
        ...additionalToolNames,
      ]);
      // Prevent recursion — sub-agents cannot delegate
      allowedToolNames.delete('delegate_task');

      const subAgentTools = allTools.filter((t) =>
        allowedToolNames.has(t.name),
      );

      // 3. Run sub-agent
      const result = await runSubAgent({
        task,
        role,
        provider,
        tools: subAgentTools,
      });

      // 4. Format result
      if (result.success) {
        return (
          `[Sub-agent (${role}) completed in ${result.iterations} iteration(s) via ${result.providerId}]\n\n` +
          result.response
        );
      } else {
        return (
          `[Sub-agent (${role}) failed after ${result.iterations} iteration(s)]\n\n` +
          `Error: ${result.response}`
        );
      }
    },
  };
}
