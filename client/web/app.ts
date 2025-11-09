import express from "express";
import cors from "cors";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { join } from "path";
import { spawn } from "child_process";

const app = express();
const PORT = process.env.PORT || 3000;
// Try multiple paths for MCP server
function findMCPServer(): string {
  const possiblePaths = [
    join(process.cwd(), "..", "mcp", "dist", "server.js"),
    join(__dirname, "..", "..", "mcp", "dist", "server.js"),
    join(process.cwd(), "mcp", "dist", "server.js"),
  ];
  
  for (const path of possiblePaths) {
    try {
      if (require("fs").existsSync(path)) {
        return path;
      }
    } catch {
      // Continue
    }
  }
  
  // Fallback to source
  return join(process.cwd(), "..", "mcp", "server.ts");
}

const MCP_SERVER_PATH = findMCPServer();

app.use(cors());
app.use(express.json());
app.use(express.static(join(__dirname, "ui")));

// Store active clients (simple in-memory store)
const clients = new Map<string, Client>();

async function getClient(sessionId: string): Promise<Client> {
  if (clients.has(sessionId)) {
    return clients.get(sessionId)!;
  }

  const isTypeScript = MCP_SERVER_PATH.endsWith(".ts");
  const command = isTypeScript ? "tsx" : "node";
  const args = isTypeScript ? [MCP_SERVER_PATH] : [MCP_SERVER_PATH];

  const transport = new StdioClientTransport({
    command,
    args,
  });

  const client = new Client(
    {
      name: "code-analyser-web",
      version: "0.1.0",
    },
    {
      capabilities: {},
    }
  );

  await client.connect(transport);
  clients.set(sessionId, client);
  return client;
}

app.post("/api/query", async (req, res) => {
  const { query, type, sessionId } = req.body;

  if (!query || !type) {
    return res.status(400).json({ error: "Query and type required" });
  }

  const sid = sessionId || "default";
  const client = await getClient(sid);

  try {
    let toolName: string;
    let args: any;

    switch (type) {
      case "code":
        toolName = "ask_code";
        args = { query };
        break;
      case "docs":
        toolName = "ask_docs";
        args = { query };
        break;
      case "both":
        toolName = "ask_combined";
        args = { query };
        break;
      default:
        return res.status(400).json({ error: "Invalid type" });
    }

    const result = await client.callTool({
      name: toolName,
      arguments: args,
    });

    if (result.isError) {
      return res.status(500).json({ error: result.content[0].text });
    }

    res.json({
      answer: result.content[0].text,
      type,
    });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/symbols", async (req, res) => {
  const { q, sessionId } = req.body;

  if (!q) {
    return res.status(400).json({ error: "Query required" });
  }

  const sid = sessionId || "default";
  const client = await getClient(sid);

  try {
    const result = await client.callTool({
      name: "search_symbols",
      arguments: { q },
    });

    if (result.isError) {
      return res.status(500).json({ error: result.content[0].text });
    }

    res.json({ result: result.content[0].text });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/trace", async (req, res) => {
  const { symbol, sessionId } = req.body;

  if (!symbol) {
    return res.status(400).json({ error: "Symbol required" });
  }

  const sid = sessionId || "default";
  const client = await getClient(sid);

  try {
    const result = await client.callTool({
      name: "trace_calls",
      arguments: { symbol },
    });

    if (result.isError) {
      return res.status(500).json({ error: result.content[0].text });
    }

    res.json({ result: result.content[0].text });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Web UI running on http://localhost:${PORT}`);
});

