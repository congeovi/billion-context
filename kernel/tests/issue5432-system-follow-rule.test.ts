import { test } from "node:test";
import assert from "node:assert/strict";
import {
  appendAnthropicUserMessages,
  coreToAnthropic,
  illegalAnthropicSystemPositions,
} from "../src/wire/anthropic.js";
import type {
  AnthropicSystemMessage,
  AnthropicWireMessage,
} from "../src/wire/anthropic.js";
import type { BiliMessage } from "../src/wire/bili-message.js";

// #5432: Anthropic's mid-conversation system rule is two-sided — the entry
// must FOLLOW a user message (or an assistant ending in a server tool result)
// AND precede an assistant or end the array. The heal used to check only the
// follower, so a fold that covered the user turn before a sidecar left it
// right after an assistant ("role 'system' must follow a 'user' message…");
// and host-injected trailing users (nudge/notes) appended after a legal
// trailing system broke the precede side ("must precede an 'assistant'
// message or end the array").

function sys(content: AnthropicSystemMessage["content"]): AnthropicSystemMessage {
  return { role: "system", content };
}

function rawSys(id: string, text: string): BiliMessage {
  return { id, role: "system", contentType: "text", text, rawAnthropicMessage: sys([{ type: "text", text }]) } as never;
}

const u = (id: string, text: string): BiliMessage => ({ id, role: "user", contentType: "text", text }) as never;
const a = (id: string, text: string): BiliMessage => ({ id, role: "assistant", contentType: "text", text }) as never;

test("#5432 fold shape: system right after an assistant is re-voiced", () => {
  // The fold covered the user turn that legitimized the sidecar.
  const rebuilt = coreToAnthropic([u("u0", "q"), a("a0", "answer"), rawSys("s1", "# Environment update"), a("a1", "next")]);
  assert.deepEqual(illegalAnthropicSystemPositions(rebuilt), []);
  assert.equal(rebuilt[2]!.role, "user");
  assert.deepEqual(rebuilt[2]!.content, [{ type: "text", text: "# Environment update" }]);
});

test("#5432 negative: system after a user and before an assistant stays verbatim", () => {
  const rebuilt = coreToAnthropic([u("u0", "q"), rawSys("s1", "reminder"), a("a0", "answer")]);
  assert.equal(rebuilt[1]!.role, "system");
  assert.deepEqual(illegalAnthropicSystemPositions(rebuilt), []);
});

test("#5432 run after an assistant: only the head is demoted, the rest is legal again", () => {
  const rebuilt = coreToAnthropic([u("u0", "q"), a("a0", "x"), rawSys("s1", "one"), rawSys("s2", "two"), a("a1", "y")]);
  assert.deepEqual(rebuilt.map((m) => m.role), ["user", "assistant", "user", "system", "assistant"]);
  assert.deepEqual(illegalAnthropicSystemPositions(rebuilt), []);
});

test("#5432 predecessor rule: user and server-tool-result assistant are legal, plain assistant is not", () => {
  const S = sys([{ type: "text", text: "s" }]);
  const after = (prev: AnthropicWireMessage): number[] =>
    illegalAnthropicSystemPositions([{ role: "user", content: "q" }, prev, S, { role: "assistant", content: "r" }]);
  assert.deepEqual(after({ role: "user", content: "q2" }), []);
  assert.deepEqual(
    after({ role: "assistant", content: [{ type: "server_tool_use", id: "s", name: "web_search", input: {} }, { type: "web_search_tool_result", tool_use_id: "s", content: [] }] } as never),
    [],
  );
  assert.deepEqual(after({ role: "assistant", content: [{ type: "text", text: "x" }] }), [2]);
  // directive-only systems are accepted anywhere
  assert.deepEqual(
    illegalAnthropicSystemPositions([
      { role: "assistant", content: "x" },
      { role: "system", content: [], output_config: { effort: "low" } } as never,
      { role: "user", content: "q" },
    ]),
    [],
  );
});

test("#5432 trailing append: host users slot in before a trailing system run", () => {
  const base: AnthropicWireMessage[] = [
    { role: "user", content: "q" },
    { role: "assistant", content: "a" },
    { role: "user", content: "tool result" },
    sys([{ type: "text", text: "The user hasn't heard from you in a while" }]),
  ];
  // control: the old plain append produced the observed 400 shape
  assert.deepEqual(illegalAnthropicSystemPositions([...base, { role: "user", content: "nudge" }]), [3]);
  const out = appendAnthropicUserMessages(base, ["nudge", "note"]);
  assert.deepEqual(illegalAnthropicSystemPositions(out), []);
  assert.deepEqual(out.map((m) => m.role), ["user", "assistant", "user", "user", "user", "system"]);
  assert.deepEqual(out[5], base[3]);
  assert.deepEqual(out.slice(3, 5), [{ role: "user", content: "nudge" }, { role: "user", content: "note" }]);
});

test("#5432 trailing append negative: no trailing system → plain append, no-op on empty", () => {
  const base: AnthropicWireMessage[] = [{ role: "user", content: "q" }, { role: "assistant", content: "a" }];
  assert.deepEqual(appendAnthropicUserMessages(base, ["n"]), [...base, { role: "user", content: "n" }]);
  assert.equal(appendAnthropicUserMessages(base, []), base);
  const onlySys: AnthropicWireMessage[] = [sys("s")];
  assert.deepEqual(appendAnthropicUserMessages(onlySys, ["n"]), [...onlySys, { role: "user", content: "n" }]);
});
