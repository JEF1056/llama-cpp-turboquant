/**
 * Utilities for parsing potentially-incomplete JSON strings during streaming.
 *
 * During streaming, JSON is built token-by-token and may be cut off mid-value.
 * parsePartialJson attempts to produce a usable tree even from incomplete input.
 */

export interface PartialJsonResult {
	/** Parsed value, or undefined if unparseable */
	value: unknown;
	/** Whether the JSON was complete and fully valid */
	complete: boolean;
}

/**
 * Parse a JSON string that may be incomplete (e.g. mid-stream).
 *
 * Strategy:
 * 1. Try JSON.parse directly.
 * 2. If that fails, attempt to repair by:
 *    - Truncating any unterminated string (avoids injecting wrong closing quotes)
 *    - Stripping dangling keys/commas
 *    - Appending missing closing brackets/braces
 * 3. Return { value: undefined } if both attempts fail.
 */
export function parsePartialJson(str: string): PartialJsonResult {
	const trimmed = str?.trim();
	if (!trimmed) return { value: undefined, complete: false };

	// Fast path: valid complete JSON
	try {
		return { value: JSON.parse(trimmed), complete: true };
	} catch {
		// fall through to repair
	}

	// Repair path
	const repaired = repairJson(trimmed);
	if (repaired !== null) {
		try {
			return { value: JSON.parse(repaired), complete: false };
		} catch {
			// fall through
		}
	}

	return { value: undefined, complete: false };
}

/**
 * Attempt to close an incomplete JSON string into parseable form.
 *
 * Uses a bracket-depth stack rather than regex so it handles nested
 * structures correctly. Never injects a closing `"` for unterminated
 * strings — instead, truncates to before the string started.
 */
function repairJson(str: string): string | null {
	const stack: ('{' | '[')[] = [];
	let inString = false;
	let stringStartPos = -1;
	let escaped = false;

	for (let i = 0; i < str.length; i++) {
		const ch = str[i];

		if (escaped) {
			escaped = false;
			continue;
		}

		if (inString) {
			if (ch === '\\') {
				escaped = true;
			} else if (ch === '"') {
				inString = false;
				stringStartPos = -1;
			}
			continue;
		}

		switch (ch) {
			case '"':
				inString = true;
				stringStartPos = i;
				break;
			case '{':
				stack.push('{');
				break;
			case '[':
				stack.push('[');
				break;
			case '}':
				if (stack.at(-1) === '{') stack.pop();
				break;
			case ']':
				if (stack.at(-1) === '[') stack.pop();
				break;
		}
	}

	// Truncate at start of any unterminated string to avoid injecting wrong closers
	let result = inString && stringStartPos >= 0 ? str.slice(0, stringStartPos) : str;

	// Strip dangling "key": (no value yet) or trailing comma
	result = result
		.trimEnd()
		.replace(/"[^"]*"\s*:\s*$/, '') // "key": with no value
		.replace(/,\s*$/, '') // trailing comma
		.trimEnd();

	if (!result) return null;

	// Close all open structures in reverse order
	for (let j = stack.length - 1; j >= 0; j--) {
		result += stack[j] === '{' ? '}' : ']';
	}

	return result;
}
