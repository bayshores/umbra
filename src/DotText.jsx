import { dotLayout } from './dot-matrix.mjs';

// Dot-matrix text (glyphs in dot-matrix.mjs). The SVG is decorative; the real text is kept for screen readers.
export default function DotText({ text, pitch = 4, className = '', title }) {
  const { lit, unlit, columns, rows } = dotLayout(text);
  const r = pitch * .38, c = n => n * pitch + pitch / 2;
  return <span className={'dot-text ' + className} title={title}>
    <svg width={columns * pitch} height={rows * pitch} viewBox={`0 0 ${columns * pitch} ${rows * pitch}`} aria-hidden="true" focusable="false">
      <g className="dot-text-off">{unlit.map(([x, y]) => <circle key={x + ',' + y} cx={c(x)} cy={c(y)} r={r}/>)}</g>
      <g className="dot-text-on">{lit.map(([x, y]) => <circle key={x + ',' + y} cx={c(x)} cy={c(y)} r={r}/>)}</g>
    </svg>
    <span className="sr-only">{text}</span>
  </span>;
}
