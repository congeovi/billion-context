import { test } from "node:test";
import assert from "node:assert/strict";

import { anthropicToCore, type AnthropicRequestBody } from "../src/wire/anthropic.js";

function turn(blocks: unknown[]): AnthropicRequestBody["messages"][number] {
  return { role: "assistant", content: blocks } as AnthropicRequestBody["messages"][number];
}

function body(...assistants: unknown[][]): AnthropicRequestBody {
  const messages: AnthropicRequestBody["messages"] = [];
  for (const blocks of assistants) {
    messages.push({ role: "user", content: "q" } as AnthropicRequestBody["messages"][number]);
    messages.push(turn(blocks));
  }
  return { model: "m", messages } as AnthropicRequestBody;
}

function ids(b: AnthropicRequestBody, type: string): string[] {
  return anthropicToCore(b).msgs
    .filter((m) => (type === "reasoning" ? m.contentType === "reasoning" : m.text === "[redacted_thinking]"))
    .map((m) => m.id);
}

test("empty-text thinking blocks get identity from their signature, not their position", () => {
  const a = { type: "thinking", thinking: "", signature: "SIG_A" };
  const b = { type: "thinking", thinking: "", signature: "SIG_B" };
  const both = ids(body([a], [b]), "reasoning");
  assert.equal(new Set(both).size, 2);
  assert.ok(both.every((id) => /^h_[0-9a-f]{16}$/.test(id)), "distinct signatures need no cluster suffix");
  const onlyB = ids(body([b]), "reasoning");
  assert.equal(onlyB[0], both[1], "dropping an earlier block must not re-key a later one");
});

test("non-empty thinking keeps its text-derived id regardless of signature", () => {
  const x = ids(body([{ type: "thinking", thinking: "plan", signature: "S1" }]), "reasoning");
  const y = ids(body([{ type: "thinking", thinking: "plan", signature: "S2" }]), "reasoning");
  assert.deepEqual(x, y);
});

test("identical empty thinking (same signature) still clusters", () => {
  const a = { type: "thinking", thinking: "", signature: "SAME" };
  const both = ids(body([a], [a]), "reasoning");
  assert.equal(both[1], `${both[0]}_1`);
});

test("redacted_thinking blocks get identity from their data", () => {
  const r1 = { type: "redacted_thinking", data: "D1" };
  const r2 = { type: "redacted_thinking", data: "D2" };
  const both = ids(body([r1], [r2]), "redacted");
  assert.equal(new Set(both).size, 2);
  assert.equal(ids(body([r2]), "redacted")[0], both[1]);
});
