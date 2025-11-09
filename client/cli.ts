#!/usr/bin/env node
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { spawn } from "child_process";
import { join } from "path";

// Try multiple paths for MCP server
function findMCPServer(): string {
  const possiblePaths = [
    join(process.cwd(), "mcp", "dist", "server.js"),
    join(__dirname, "..", "mcp", "dist", "server.js"),
    join(process.cwd(), "..", "mcp", "dist", "server.js"),
  ];
  
  for (const path of possiblePaths) {
    if (require("fs").existsSync(path)) {
      return path;
    }
  }
  
  // Fallback to source if dist doesn't exist
  return join(process.cwd(), "mcp", "server.ts");
}

const MCP_SERVER_PATH = findMCPServer();

async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0 || args[0] === "help" || args[0] === "--help" || args[0] === "-h") {
    printHelp();
    process.exit(0);
  }

  const cmd = args[0];
  const commandArgs = args.slice(1);

  // Determine command based on file extension
  const isTypeScript = MCP_SERVER_PATH.endsWith(".ts");
  const mcpCommand = isTypeScript ? "tsx" : "node";
  const mcpArgs = isTypeScript ? [MCP_SERVER_PATH] : [MCP_SERVER_PATH];

  const transport = new StdioClientTransport({
    command: mcpCommand,
    args: mcpArgs,
  });

  const client = new Client(
    {
      name: "code-analyser-cli",
      version: "0.1.0",
    },
    {
      capabilities: {},
    }
  );

  try {
    await client.connect(transport);

    switch (cmd) {
      case "ask:code":
        await handleAskCode(client, commandArgs);
        break;
      case "ask:docs":
        await handleAskDocs(client, commandArgs);
        break;
      case "ask:both":
        await handleAskBoth(client, commandArgs);
        break;
      case "symbols":
        await handleSymbols(client, commandArgs);
        break;
      case "trace":
        await handleTrace(client, commandArgs);
        break;
      case "reindex":
        await handleReindex(client, commandArgs);
        break;
      default:
        console.error(`Unknown command: ${cmd}`);
        printHelp();
        process.exit(1);
    }
  } catch (error: any) {
    console.error("Error:", error.message);
    process.exit(1);
  } finally {
    await client.close();
  }
}

async function handleAskCode(client: Client, args: string[]) {
  if (args.length === 0) {
    console.error("Error: Query required");
    process.exit(1);
  }
  const query = args.join(" ");
  
  const result = await client.callTool({
    name: "ask_code",
    arguments: { query },
  });

  if (result.isError) {
    console.error("Error:", result.content[0].text);
    process.exit(1);
  }

  console.log("\n" + result.content[0].text + "\n");
}

async function handleAskDocs(client: Client, args: string[]) {
  if (args.length === 0) {
    console.error("Error: Query required");
    process.exit(1);
  }
  const query = args.join(" ");
  
  const result = await client.callTool({
    name: "ask_docs",
    arguments: { query },
  });

  if (result.isError) {
    console.error("Error:", result.content[0].text);
    process.exit(1);
  }

  console.log("\n" + result.content[0].text + "\n");
}

async function handleAskBoth(client: Client, args: string[]) {
  if (args.length === 0) {
    console.error("Error: Query required");
    process.exit(1);
  }
  const query = args.join(" ");
  
  const result = await client.callTool({
    name: "ask_combined",
    arguments: { query },
  });

  if (result.isError) {
    console.error("Error:", result.content[0].text);
    process.exit(1);
  }

  console.log("\n" + result.content[0].text + "\n");
}

async function handleSymbols(client: Client, args: string[]) {
  if (args.length === 0) {
    console.error("Error: Symbol name required");
    process.exit(1);
  }
  const symbol = args.join(" ");
  
  const result = await client.callTool({
    name: "search_symbols",
    arguments: { q: symbol },
  });

  if (result.isError) {
    console.error("Error:", result.content[0].text);
    process.exit(1);
  }

  console.log("\n" + result.content[0].text + "\n");
}

async function handleTrace(client: Client, args: string[]) {
  if (args.length === 0) {
    console.error("Error: Symbol name required");
    process.exit(1);
  }
  const symbol = args.join(" ");
  
  const result = await client.callTool({
    name: "trace_calls",
    arguments: { symbol },
  });

  if (result.isError) {
    console.error("Error:", result.content[0].text);
    process.exit(1);
  }

  console.log("\n" + result.content[0].text + "\n");
}

async function handleReindex(client: Client, args: string[]) {
  let target: "code" | "docs" | "all" = "all";
  
  const onlyIndex = args.indexOf("--only");
  if (onlyIndex !== -1 && args[onlyIndex + 1]) {
    const onlyValue = args[onlyIndex + 1];
    if (onlyValue === "code" || onlyValue === "docs") {
      target = onlyValue;
    }
  }
  
  console.log(`Reindexing ${target}...\n`);
  
  const result = await client.callTool({
    name: "update_index",
    arguments: { target },
  });

  if (result.isError) {
    console.error("Error:", result.content[0].text);
    process.exit(1);
  }

  console.log(result.content[0].text);
}

function printHelp() {
  console.log(`
Code Analyser CLI

Usage:
  cli <command> [arguments]

Commands:
  ask:code <query>        Ask questions about code
  ask:docs <query>        Ask questions about documents
  ask:both <query>        Ask questions using both code and docs
  symbols <name>          Search for symbols in codebase
  trace <symbol>          Trace call graph for a symbol
  reindex [--only code|docs]  Rebuild the index

Examples:
  cli ask:code "Where is user session validated?"
  cli ask:docs "What is the retry policy?"
  cli ask:both "Does the code match the architecture docs?"
  cli symbols "AuthManager"
  cli trace "processPayment"
  cli reindex --only docs
`);
}

main().catch(console.error);

