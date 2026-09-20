import Editor, { type OnMount } from "@monaco-editor/react";
import { useRef } from "react";

import { EDITOR_OPTIONS, THEME } from "../lib/monaco";
import { WIDE, useMediaQuery } from "../lib/useMediaQuery";

type Props = {
  value: string;
  onChange: (next: string) => void;
  onRun: () => void;
  onCheck: () => void;
  readOnly?: boolean;
};

/**
 * The Python editor. Cmd or Ctrl with Return runs the file; adding Shift runs
 * the checks, which matches what the two buttons above it do.
 */
export default function CodeEditor({
  value,
  onChange,
  onRun,
  onCheck,
  readOnly = false,
}: Props) {
  // A phone fits about 35 characters across, so unwrapped code spends most of
  // its time off the right edge. Wide screens keep the lines as written.
  const wide = useMediaQuery(WIDE);

  // The keyboard commands are registered once, against a ref, so that changing
  // the handler between renders does not re-register them.
  const run = useRef(onRun);
  const check = useRef(onCheck);
  run.current = onRun;
  check.current = onCheck;

  const mount: OnMount = (editor, monaco) => {
    monaco.editor.setTheme(THEME);
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () =>
      run.current(),
    );
    editor.addCommand(
      monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter,
      () => check.current(),
    );
    editor.focus();
  };

  return (
    <Editor
      language="python"
      theme={THEME}
      value={value}
      onChange={(next) => onChange(next ?? "")}
      onMount={mount}
      options={{
        ...EDITOR_OPTIONS,
        readOnly,
        wordWrap: wide ? "off" : "on",
        // The 88-column ruler is meaningless once the lines wrap.
        rulers: wide ? EDITOR_OPTIONS.rulers : [],
        lineNumbersMinChars: wide ? 5 : 3,
        folding: wide,
      }}
      loading={
        <div className="flex h-full items-center justify-center text-xs text-ink-600">
          Loading the editor…
        </div>
      }
    />
  );
}
