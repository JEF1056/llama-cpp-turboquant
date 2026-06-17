/**
 * Rehype plugin to enhance bare URLs that the model writes as plain text.
 *
 * GitHub Flavored Markdown autolinks bare URLs into anchor elements whose text
 * equals the href. This plugin upgrades those autolinks:
 * - Image URLs (e.g. https://example.com/cat.png) are replaced with an inline
 *   <img> element so the image is shown directly.
 * - Other http(s) URLs get a `.link-preview` placeholder appended after their
 *   block, which is hydrated into a preview card after render.
 *
 * Deliberately-labelled markdown links (e.g. [text](url)) are left untouched,
 * since their anchor text differs from the href.
 */

import type { Root, Element, ElementContent } from 'hast';
import { visit } from 'unist-util-visit';
import { BARE_IMAGE_URL_REGEX, LINK_PREVIEW_CLASS, LINK_PREVIEW_URL_ATTR } from '$lib/constants';

/**
 * Whether the href is an absolute http(s) URL.
 */
function isHttpUrl(href: string): boolean {
	return /^https?:\/\//i.test(href);
}

/**
 * Returns the anchor's text when it is a bare autolink (single text child whose
 * value matches the href), otherwise null.
 */
function bareLinkText(node: Element, href: string): string | null {
	if (node.children.length !== 1) return null;

	const child = node.children[0];
	if (child.type !== 'text') return null;

	const text = child.value.trim();
	if (!text) return null;

	const normalizedHref = href.replace(/\/$/, '');
	const normalizedText = text.replace(/\/$/, '');

	const isBare =
		href === text ||
		normalizedHref === normalizedText ||
		href === `http://${text}` ||
		href === `https://${text}`;

	return isBare ? text : null;
}

/**
 * Whether the href points to a renderable image based on its path extension.
 */
function isImageHref(href: string): boolean {
	try {
		return BARE_IMAGE_URL_REGEX.test(new URL(href).pathname);
	} catch {
		return BARE_IMAGE_URL_REGEX.test(href.split(/[?#]/)[0]);
	}
}

/**
 * Builds an <img> hast element for an image URL.
 */
function createImageNode(href: string): Element {
	return {
		type: 'element',
		tagName: 'img',
		properties: { src: href, alt: '', loading: 'lazy' },
		children: []
	};
}

/**
 * Builds the link-preview placeholder hast element for a URL.
 */
function createPreviewPlaceholder(url: string): Element {
	return {
		type: 'element',
		tagName: 'div',
		properties: { className: [LINK_PREVIEW_CLASS], [LINK_PREVIEW_URL_ATTR]: url },
		children: []
	};
}

export function rehypeEnhanceBareUrls() {
	return (tree: Root) => {
		const previewUrls: string[] = [];
		const seen = new Set<string>();

		visit(tree, 'element', (node: Element, index, parent) => {
			if (node.tagName !== 'a') return;

			const href = node.properties?.href;
			if (typeof href !== 'string' || !isHttpUrl(href)) return;

			const text = bareLinkText(node, href);
			if (text === null) return;

			if (isImageHref(href)) {
				if (parent && typeof index === 'number') {
					parent.children[index] = createImageNode(href) as ElementContent;
				}
				return;
			}

			if (!seen.has(href)) {
				seen.add(href);
				previewUrls.push(href);
			}
		});

		for (const url of previewUrls) {
			tree.children.push(createPreviewPlaceholder(url));
		}
	};
}
