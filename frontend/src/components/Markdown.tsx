import { memo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

const REMARK = [remarkGfm, remarkMath];
const REHYPE = [
  [rehypeKatex, { throwOnError: false, strict: false }],
  [rehypeHighlight, { detect: true, ignoreMissing: true }],
] as never;

/**
 * Renders a lesson body: GitHub-flavoured Markdown, KaTeX for the maths, and
 * highlight.js for the code blocks. Memoised because a lesson body is long and
 * the pane around it rerenders on every keystroke in the editor.
 */
function Markdown({ children }: { children: string }) {
  return (
    <div className="lesson">
      <ReactMarkdown remarkPlugins={REMARK} rehypePlugins={REHYPE}>
        {children}
      </ReactMarkdown>
    </div>
  );
}

export default memo(Markdown);
