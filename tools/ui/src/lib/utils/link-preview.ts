/**
 * Client-side link unfurling for markdown link previews.
 *
 * Fetches a URL's HTML through llama-server's CORS proxy (the only way to read
 * cross-origin page markup from the browser) and extracts OpenGraph / Twitter /
 * standard metadata to build a preview card. Results (including failures) are
 * cached per URL so repeated renders during streaming do not refetch.
 */

import { serverStore } from '$lib/stores/server.svelte';
import { buildProxiedUrl } from './cors-proxy';

export interface LinkPreviewData {
	url: string;
	title?: string;
	description?: string;
	image?: string;
	siteName?: string;
}

const cache = new Map<string, Promise<LinkPreviewData | null>>();

/**
 * Reads the first matching <meta> content for any of the given property/name keys.
 */
function readMeta(doc: Document, keys: string[]): string | undefined {
	for (const key of keys) {
		const el =
			doc.querySelector(`meta[property="${key}"]`) ?? doc.querySelector(`meta[name="${key}"]`);
		const content = el?.getAttribute('content')?.trim();
		if (content) return content;
	}
	return undefined;
}

/**
 * Resolves a possibly-relative image URL against the page URL, keeping only http(s).
 */
function resolveImage(image: string | undefined, pageUrl: string): string | undefined {
	if (!image) return undefined;
	try {
		const resolved = new URL(image, pageUrl);
		if (resolved.protocol !== 'http:' && resolved.protocol !== 'https:') return undefined;
		return resolved.href;
	} catch {
		return undefined;
	}
}

/**
 * Parses preview metadata out of an HTML document string.
 */
function parseMetadata(html: string, url: string): LinkPreviewData | null {
	const doc = new DOMParser().parseFromString(html, 'text/html');

	const title =
		readMeta(doc, ['og:title', 'twitter:title']) ?? doc.querySelector('title')?.textContent?.trim();
	const description = readMeta(doc, ['og:description', 'twitter:description', 'description']);
	const image = resolveImage(
		readMeta(doc, ['og:image', 'twitter:image', 'twitter:image:src']),
		url
	);
	const siteName = readMeta(doc, ['og:site_name']) ?? safeHostname(url);

	if (!title && !description && !image) return null;

	return { url, title, description, image, siteName };
}

/**
 * Returns the hostname for a URL, or undefined when it cannot be parsed.
 */
function safeHostname(url: string): string | undefined {
	try {
		return new URL(url).hostname;
	} catch {
		return undefined;
	}
}

/**
 * Fetches and parses link preview metadata for a URL through the CORS proxy.
 */
async function unfurl(url: string): Promise<LinkPreviewData | null> {
	if (!serverStore.props?.cors_proxy_enabled) return null;

	try {
		const res = await fetch(buildProxiedUrl(url), { method: 'GET' });
		if (!res.ok) return null;

		const contentType = res.headers.get('content-type') ?? '';
		if (!contentType.includes('text/html')) return null;

		const html = await res.text();
		return parseMetadata(html, url);
	} catch {
		return null;
	}
}

/**
 * Returns link preview metadata for a URL, caching the result per URL.
 */
export function fetchLinkPreview(url: string): Promise<LinkPreviewData | null> {
	const cached = cache.get(url);
	if (cached) return cached;

	const pending = unfurl(url);
	cache.set(url, pending);
	return pending;
}
