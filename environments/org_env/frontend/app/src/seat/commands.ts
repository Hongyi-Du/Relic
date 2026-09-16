// Slash command parser for WorkingAgent quick actions
// Transforms shorthand commands into natural language for the agent

export type Command = {
  type: string;
  rawInput: string;
  naturalLanguage: string;
  params?: Record<string, string>;
};

const COMMANDS = {
  draft: {
    syntax: "/draft <action> <target>",
    example: "/draft reply @Alice about the API bug",
    description: "Draft an action (reply, proposal, task, etc.)",
  },
  read: {
    syntax: "/read <object_id>",
    example: "/read T-123",
    description: "Read and summarize an object",
  },
  search: {
    syntax: "/search <query>",
    example: "/search experiments status:running",
    description: "Search tasks, messages, or code",
  },
  status: {
    syntax: "/status",
    example: "/status",
    description: "Show my current work status",
  },
  help: {
    syntax: "/help",
    example: "/help",
    description: "Show available commands",
  },
};

export function parseCommand(input: string): Command | null {
  const trimmed = input.trim();
  if (!trimmed.startsWith("/")) return null;

  const parts = trimmed.slice(1).split(/\s+/);
  const cmd = parts[0].toLowerCase();
  const args = parts.slice(1).join(" ");

  switch (cmd) {
    case "draft":
      return {
        type: "draft",
        rawInput: input,
        naturalLanguage: `Draft a ${args}`,
        params: { action: args },
      };

    case "read":
      return {
        type: "read",
        rawInput: input,
        naturalLanguage: `Read ${args} and summarize it for me`,
        params: { object_id: args },
      };

    case "search":
      return {
        type: "search",
        rawInput: input,
        naturalLanguage: `Search for ${args}`,
        params: { query: args },
      };

    case "status":
      return {
        type: "status",
        rawInput: input,
        naturalLanguage: "Show me my current work status and what needs attention",
        params: {},
      };

    case "help":
      return {
        type: "help",
        rawInput: input,
        naturalLanguage: generateHelpText(),
        params: {},
      };

    default:
      return null;
  }
}

function generateHelpText(): string {
  const lines = ["Available commands:\n"];
  for (const [name, info] of Object.entries(COMMANDS)) {
    lines.push(`/${name} — ${info.description}`);
    lines.push(`  Example: ${info.example}\n`);
  }
  return lines.join("\n");
}

export function getCommandSuggestions(input: string): string[] {
  if (!input.startsWith("/")) return [];

  const partial = input.slice(1).toLowerCase();
  if (partial === "") {
    return Object.keys(COMMANDS).map((c) => `/${c}`);
  }

  return Object.keys(COMMANDS)
    .filter((c) => c.startsWith(partial))
    .map((c) => `/${c}`);
}

export { COMMANDS };
