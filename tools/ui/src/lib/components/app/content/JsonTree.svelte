<script lang="ts">
	import { SvelteMap } from 'svelte/reactivity';
	import { parsePartialJson } from '$lib/utils/json-partial';
	import JsonTreeNode from './JsonTreeNode.svelte';

	interface Props {
		/** Raw JSON string — may be incomplete during streaming */
		json: string;
		/** When true, shows a streaming indicator dot */
		partial?: boolean;
		/**
		 * Stable identity for this call (e.g. toolName + sectionIndex).
		 * When it changes, the collapse-state map is cleared so collapsed
		 * nodes from a previous call don't bleed into the next one.
		 */
		callId?: string;
		maxHeight?: string;
		class?: string;
	}

	let { json, partial = false, callId = '', maxHeight = '20rem', class: className = '' }: Props =
		$props();

	/**
	 * Shared reactive collapse state map: JSON path → isCollapsed.
	 * Owned here at the root; passed by reference to all JsonTreeNode descendants.
	 */
	const collapsed = new SvelteMap<string, boolean>();

	let prevCallId = $state('');
	$effect(() => {
		if (callId !== prevCallId) {
			prevCallId = callId;
			collapsed.clear();
		}
	});

	const parsed = $derived(parsePartialJson(json));
</script>

<div
	class="json-tree {className}"
	style="max-height: {maxHeight};"
	role="tree"
	aria-label="JSON value"
>
	{#if parsed.value !== undefined}
		<JsonTreeNode value={parsed.value} path="root" depth={0} {collapsed} />
	{:else if json}
		<!-- Unparseable — fall back to raw text so nothing is lost -->
		<span class="json-raw">{json}</span>
	{:else}
		<span class="json-empty">(empty)</span>
	{/if}

	{#if partial}
		<span class="json-streaming" aria-label="Streaming…">
			<span class="json-dot"></span>
		</span>
	{/if}
</div>

<style>
	.json-tree {
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas,
			'Liberation Mono', Menlo, monospace;
		font-size: 0.75rem;
		line-height: 1.4;
		/* overflow-x must be hidden (not auto) so the container constrains child widths,
		   which lets overflow-wrap/word-break actually trigger on string values. */
		overflow-y: auto;
		overflow-x: hidden;
		padding: 0.5rem;
		color: var(--foreground);
	}

	.json-raw {
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		color: var(--muted-foreground);
	}

	.json-empty {
		color: var(--muted-foreground);
		font-style: italic;
	}

	.json-streaming {
		display: inline-flex;
		align-items: center;
		padding: 0.25rem 0;
	}

	.json-dot {
		display: inline-block;
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: var(--muted-foreground);
		animation: json-pulse 1s ease-in-out infinite;
	}

	@keyframes json-pulse {
		0%,
		100% {
			opacity: 0.25;
		}
		50% {
			opacity: 1;
		}
	}
</style>
