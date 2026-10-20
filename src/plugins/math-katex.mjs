// 构建期将 $...$ / $$...$$ 渲染为 KaTeX HTML（配合 features.math 使用）
import katex from 'katex';

function render(value, displayMode) {
  return katex.renderToString(value, {
    displayMode,
    throwOnError: false,
    strict: 'ignore',
    trust: false,
    output: 'htmlAndMathml',
  });
}

/** @type {import('satteri').MdastPluginDefinition} */
export const mathKatex = {
  name: 'math-katex',
  math(node) {
    return { type: 'html', value: `<div class="math-block">${render(node.value, true)}</div>` };
  },
  inlineMath(node) {
    return { type: 'html', value: render(node.value, false) };
  },
};
