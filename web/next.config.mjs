/** Static export so the same UI ships to GitHub Pages; the API is optional at runtime. */
const basePath = process.env.PAGES_BASE_PATH || "";
export default { output: "export", basePath, images: { unoptimized: true }, env: { NEXT_PUBLIC_BASE_PATH: basePath } };
