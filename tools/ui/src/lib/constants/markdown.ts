export const IMAGE_NOT_ERROR_BOUND_SELECTOR = 'img:not([data-error-bound])';
export const DATA_ERROR_BOUND_ATTR = 'errorBound';
export const DATA_ERROR_HANDLED_ATTR = 'errorHandled';
export const BOOL_TRUE_STRING = 'true';

// Matches URLs whose path points to a renderable image, used to turn bare
// image URLs written in plain text into inline <img> elements.
export const BARE_IMAGE_URL_REGEX = /\.(png|jpe?g|gif|webp|avif|bmp|svg)$/i;

// Link preview placeholder emitted by the rehype bare-url plugin and hydrated
// into a preview card after render.
export const LINK_PREVIEW_CLASS = 'link-preview';
export const LINK_PREVIEW_URL_ATTR = 'data-link-preview-url';
export const LINK_PREVIEW_READY_ATTR = 'data-link-preview-ready';
export const LINK_PREVIEW_NOT_READY_SELECTOR = '.link-preview:not([data-link-preview-ready])';
