<script lang="ts">
	import type { SvelteMap } from 'svelte/reactivity';
	import ChevronRightIcon from '@lucide/svelte/icons/chevron-right';
	import ChevronDownIcon from '@lucide/svelte/icons/chevron-down';
	// Recursive self-import — Vite/Rollup resolves this correctly at bundle time
	import JsonTreeNode from './JsonTreeNode.svelte';

	interface Props {
		value: unknown;
		/** Dot/bracket path string used as key in the shared collapse map */
		path: string;
		/** Nesting depth — used for auto-collapse heuristic only, indent is CSS-based */
		depth: number;
		/** Shared reactive map owned by JsonTree; mutations trigger re-renders */
		collapsed: SvelteMap<string, boolean>;
		/** Present when this node is a child of an object or array */
		keyName?: string | number;
	}

	let { value, path, depth, collapsed, keyName }: Props = $props();

	type JsonType = 'null' | 'boolean' | 'number' | 'string' | 'array' | 'object';

	function getType(v: unknown): JsonType {
		if (v === null) return 'null';
		if (typeof v === 'boolean') return 'boolean';
		if (typeof v === 'number') return 'number';
		if (typeof v === 'string') return 'string';
		if (Array.isArray(v)) return 'array';
		return 'object';
	}

	const type = $derived(getType(value));
	const isCollapsible = $derived(type === 'array' || type === 'object');

	const entries = $derived<[string | number, unknown][]>(
		type === 'array'
			? (value as unknown[]).map((v, i) => [i, v])
			: type === 'object'
				? Object.entries(value as Record<string, unknown>)
				: []
	);

	const childCount = $derived(entries.length);

	/** Auto-collapse objects/arrays that have many children, except at the root */
	function defaultCollapsed(): boolean {
		if (depth === 0) return false;
		return childCount > 10;
	}

	const isNodeCollapsed = $derived(
		collapsed.has(path) ? (collapsed.get(path) as boolean) : defaultCollapsed()
	);

	function toggle() {
		collapsed.set(path, !isNodeCollapsed);
	}

	function childPath(key: string | number): string {
		return typeof key === 'number' ? `${path}[${key}]` : `${path}.${key}`;
	}

	const openBracket = $derived(type === 'array' ? '[' : '{');
	const closeBracket = $derived(type === 'array' ? ']' : '}');
	const collapsedSummary = $derived(
		type === 'array'
			? `[ ${childCount} ${childCount === 1 ? 'item' : 'items'} ]`
			: `{ ${childCount} ${childCount === 1 ? 'key' : 'keys'} }`
	);
</script>

{#if !isCollapsible}
	<!-- Leaf node: renders as a single line -->
	<div class="json-node json-leaf">
		{#if keyName !== undefined}
			<span class="json-key"
				>{typeof keyName === 'number' ? keyName : `"${keyName}"`}</span
			><span class="json-punct">:&nbsp;</span>
		{/if}
		{#if type === 'null'}
			<span class="json-null">null</span>
		{:else if type === 'boolean'}
			<span class="json-bool">{String(value)}</span>
		{:else if type === 'number'}
			<span class="json-num">{value as number}</span>
		{:else}
			<span class="json-str">"<span class="json-str-inner">{value as string}</span>"</span>
		{/if}
	</div>
{:else}
	<!-- Collapsible node: opening + optional children + closing bracket -->
	<div class="json-node json-collapsible-header">
		<button
			type="button"
			class="json-toggle"
			onclick={toggle}
			aria-label={isNodeCollapsed ? 'Expand node' : 'Collapse node'}
		>
			{#if isNodeCollapsed}
				<ChevronRightIcon class="json-chevron" />
			{:else}
				<ChevronDownIcon class="json-chevron" />
			{/if}
		</button>

		{#if keyName !== undefined}
			<span class="json-key"
				>{typeof keyName === 'number' ? keyName : `"${keyName}"`}</span
			><span class="json-punct">:&nbsp;</span>
		{/if}

		{#if isNodeCollapsed}
			<button type="button" class="json-collapsed-preview" onclick={toggle}>
				{collapsedSummary}
			</button>
		{:else}
			<span class="json-punct">{openBracket}</span>
		{/if}
	</div>

	{#if !isNodeCollapsed}
		<div class="json-children">
			{#each entries as [k, v] (k)}
				<JsonTreeNode value={v} path={childPath(k)} depth={depth + 1} {collapsed} keyName={k} />
			{/each}
		</div>
		<div class="json-node json-closing">
			<span class="json-punct">{closeBracket}</span>
		</div>
	{/if}
{/if}

<style>
	.json-node {
		display: flex;
		align-items: baseline;
		gap: 0;
		min-height: 1.4em;
		line-height: 1.4;
		/* Allow flex items to shrink below their content width so text can wrap */
		min-width: 0;
	}

	.json-children {
		padding-left: 1.25rem;
	}

	/* Toggle button */
	.json-toggle {
		display: inline-flex;
		align-items: center;
		background: transparent;
		border: none;
		padding: 0;
		cursor: pointer;
		color: var(--muted-foreground);
		flex-shrink: 0;
		margin-right: 0.125rem;
	}

	.json-toggle:hover {
		color: var(--foreground);
	}

	:global(.json-chevron) {
		width: 0.75rem;
		height: 0.75rem;
	}

	/* Collapsed summary button */
	.json-collapsed-preview {
		background: transparent;
		border: none;
		padding: 0;
		cursor: pointer;
		color: var(--muted-foreground);
		font-family: inherit;
		font-size: inherit;
	}

	.json-collapsed-preview:hover {
		color: var(--foreground);
		text-decoration: underline;
	}

	/* Token colors */
	.json-key {
		color: hsl(215 70% 55%);
	}

	:global(.dark) .json-key {
		color: hsl(210 80% 70%);
	}

	.json-punct {
		color: var(--muted-foreground);
	}

	.json-str {
		color: hsl(20 80% 45%);
		/* Flex item must be able to shrink so the text inside can wrap */
		min-width: 0;
		flex-shrink: 1;
	}

	:global(.dark) .json-str {
		color: hsl(35 85% 65%);
	}

	.json-str-inner {
		/* Prefer breaking at word boundaries; fall back to any character for
		   URLs, base64, and other strings with no natural break points */
		overflow-wrap: anywhere;
		word-break: break-word;
	}

	.json-num {
		color: hsl(160 55% 35%);
	}

	:global(.dark) .json-num {
		color: hsl(155 60% 55%);
	}

	.json-bool {
		color: hsl(270 60% 50%);
	}

	:global(.dark) .json-bool {
		color: hsl(270 70% 70%);
	}

	.json-null {
		color: var(--muted-foreground);
		font-style: italic;
	}
</style>
