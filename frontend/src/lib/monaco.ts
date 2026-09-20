/**
 * Monaco, bundled with the app instead of fetched from a CDN at run time.
 *
 * `@monaco-editor/react` loads the editor from jsdelivr unless it is handed an
 * instance. The app is behind a login on a private host, so it builds a lean
 * instance with the features and the one language the exercises need, gives it
 * the app's colours, and registers it with the loader.
 */

import { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor/editor/editor.api";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";

// Editor features. The full set adds a diff editor, a colour picker, code
// lenses, and other machinery that has no provider here.
import "monaco-editor/features/codicon/register";
import "monaco-editor/features/bracketMatching/register";
import "monaco-editor/features/caretOperations/register";
import "monaco-editor/features/clipboard/register";
import "monaco-editor/features/comment/register";
import "monaco-editor/features/contextmenu/register";
import "monaco-editor/features/cursorUndo/register";
import "monaco-editor/features/find/register";
import "monaco-editor/features/folding/register";
import "monaco-editor/features/indentation/register";
import "monaco-editor/features/lineSelection/register";
import "monaco-editor/features/linesOperations/register";
import "monaco-editor/features/longLinesHelper/register";
import "monaco-editor/features/multicursor/register";
import "monaco-editor/features/smartSelect/register";
import "monaco-editor/features/suggest/register";
import "monaco-editor/features/tokenization/register";
import "monaco-editor/features/wordHighlighter/register";
import "monaco-editor/features/wordOperations/register";
import "monaco-editor/features/wordPartOperations/register";

// The one language the exercises are written in.
import "monaco-editor/languages/definitions/python/register";

self.MonacoEnvironment = {
  // Python has no language service worker of its own, so the base editor
  // worker is the only one this app needs.
  getWorker: () => new EditorWorker(),
};

export const THEME = "weights-and-wires";

monaco.editor.defineTheme(THEME, {
  base: "vs-dark",
  inherit: true,
  rules: [
    { token: "", foreground: "cdd3dd" },
    { token: "keyword", foreground: "7aa2ff" },
    { token: "string", foreground: "8fd6a0" },
    { token: "string.escape", foreground: "b4ecc2" },
    { token: "comment", foreground: "6b7488", fontStyle: "italic" },
    { token: "number", foreground: "f0a637" },
    { token: "type", foreground: "74d0e0" },
    { token: "delimiter", foreground: "7b8496" },
    { token: "identifier", foreground: "cdd3dd" },
  ],
  colors: {
    "editor.background": "#0d0f14",
    "editor.foreground": "#cdd3dd",
    "editorCursor.foreground": "#4f8cff",
    "editor.lineHighlightBackground": "#12151c",
    "editor.lineHighlightBorder": "#12151c00",
    "editor.selectionBackground": "#4f8cff33",
    "editor.inactiveSelectionBackground": "#4f8cff1a",
    "editor.selectionHighlightBackground": "#4f8cff1a",
    "editor.wordHighlightBackground": "#4f8cff1f",
    "editor.findMatchBackground": "#f0a63755",
    "editor.findMatchHighlightBackground": "#f0a63726",
    "editorLineNumber.foreground": "#39404f",
    "editorLineNumber.activeForeground": "#7b8496",
    "editorIndentGuide.background1": "#1a1e27",
    "editorIndentGuide.activeBackground1": "#39404f",
    "editorWhitespace.foreground": "#252a36",
    "editorBracketMatch.background": "#4f8cff1f",
    "editorBracketMatch.border": "#4f8cff80",
    "editorBracketHighlight.foreground1": "#7b8496",
    "editorBracketHighlight.foreground2": "#7aa2ff",
    "editorBracketHighlight.foreground3": "#8fd6a0",
    "editorGutter.background": "#0d0f14",
    "editorWidget.background": "#12151c",
    "editorWidget.border": "#252a36",
    "editorSuggestWidget.background": "#12151c",
    "editorSuggestWidget.border": "#252a36",
    "editorSuggestWidget.selectedBackground": "#1a1e27",
    "editorSuggestWidget.highlightForeground": "#74a5ff",
    "editorHoverWidget.background": "#12151c",
    "editorHoverWidget.border": "#252a36",
    "editorRuler.foreground": "#1a1e27",
    focusBorder: "#4f8cff",
    "scrollbar.shadow": "#00000000",
    "scrollbarSlider.background": "#252a3699",
    "scrollbarSlider.hoverBackground": "#39404f99",
    "scrollbarSlider.activeBackground": "#39404fcc",
    "editorOverviewRuler.border": "#00000000",
  },
});

loader.config({ monaco });

export const EDITOR_OPTIONS: monaco.editor.IStandaloneEditorConstructionOptions =
  {
    fontFamily: '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 13.5,
    lineHeight: 21,
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    smoothScrolling: true,
    cursorBlinking: "smooth",
    renderLineHighlight: "line",
    padding: { top: 14, bottom: 14 },
    tabSize: 4,
    insertSpaces: true,
    rulers: [88],
    bracketPairColorization: { enabled: true },
    automaticLayout: true,
    scrollbar: { verticalScrollbarSize: 10, horizontalScrollbarSize: 10 },
    // The course is Python, and the reader is not writing a library here.
    quickSuggestions: { other: true, comments: false, strings: false },
    suggestOnTriggerCharacters: true,
    wordBasedSuggestions: "currentDocument",
  };

export { monaco };
