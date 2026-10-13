import { useRef } from 'react';
import { Truck, Eye } from '@phosphor-icons/react';

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const metres = n => Number(n).toFixed(2);

// One axis along the lane, from the crossing (0 m) to the recorded camera. The crossing, the current vehicle, the proposed
// vehicle and the driver sit at the lane-axis distances the comparison uses. The proposed vehicle and the driver can be
// dragged, or moved with the arrow keys (Shift for 1 m steps), within the comparison's limits.
export default function LaneRuler({ limits, setback, baselineSetback, driverDistance, baselineLabel, onSetback, onDriver }) {
  const track = useRef(null), latest = useRef({});
  // Arrow keys step from the newest value, even when several key presses arrive before the next render.
  latest.current = { proposed: setback, driver: driverDistance };
  const span = limits.captureDistance;
  const at = value => `${clamp(value / span * 100, 0, 100)}%`;
  const valueAt = clientX => {
    const box = track.current.getBoundingClientRect();
    return (clientX - box.left) / box.width * span;
  };
  const drag = (onChange, min, max) => event => {
    if (event.button !== 0) return;
    const handle = event.currentTarget;
    handle.setPointerCapture(event.pointerId);
    handle.focus({ focusVisible: false });
    const move = e => onChange(clamp(valueAt(e.clientX), min, max));
    const stop = () => {
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', stop);
      handle.removeEventListener('pointercancel', stop);
      handle.classList.remove('lane-handle--dragging');
    };
    handle.classList.add('lane-handle--dragging');
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', stop);
    handle.addEventListener('pointercancel', stop);
    event.preventDefault();
  };
  const keys = (name, onChange, min, max) => event => {
    const value = Number(latest.current[name]), step = event.shiftKey ? 1 : 0.1;
    const next = { ArrowLeft: value - step, ArrowDown: value - step, ArrowRight: value + step, ArrowUp: value + step, Home: min, End: max }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    latest.current[name] = clamp(next, min, max);
    onChange(latest.current[name]);
  };
  const ticks = Array.from({ length: Math.floor(span) + 1 }, (_, m) => m);
  const handle = (name, label, value, min, max, onChange, Icon) => <div className={'lane-handle lane-handle--' + name} role="slider" tabIndex={0}
    aria-label={label} aria-valuemin={Number(metres(min))} aria-valuemax={Number(metres(max))} aria-valuenow={Number(metres(value))} aria-valuetext={metres(value) + ' metres from the crossing'}
    style={{ left: at(value) }} onPointerDown={drag(onChange, min, max)} onKeyDown={keys(name, onChange, min, max)}>
    <span className="lane-value">{metres(value)} m</span>
    <span className="lane-chip"><Icon size={14} weight="bold" aria-hidden="true"/></span>
  </div>;
  return <div className="lane-ruler">
    <div className="lane-track" ref={track}>
      <span className="lane-reach" style={{ left: at(limits.minimumSetback), width: `calc(${at(limits.maximumSetback)} - ${at(limits.minimumSetback)})` }} aria-hidden="true"/>
      {ticks.map(m => <span key={m} className={'lane-tick' + (m % 5 === 0 ? ' lane-tick--major' : '')} style={{ left: at(m) }} aria-hidden="true">{m % 5 === 0 && <span>{m} m</span>}</span>)}
      <span className="lane-crossing" style={{ left: at(0) }} title="Crossing"><span className="sr-only">Crossing at 0 metres</span></span>
      {Number.isFinite(baselineSetback) && <span className="lane-fixed lane-fixed--baseline" style={{ left: at(baselineSetback) }} title={`${baselineLabel}, ${metres(baselineSetback)} m`}>
        <Truck size={13} aria-hidden="true"/><span className="sr-only">{baselineLabel} at {metres(baselineSetback)} metres</span></span>}
      {handle('proposed', 'Proposed vehicle distance from crossing', setback, limits.minimumSetback, limits.maximumSetback, onSetback, Truck)}
      {handle('driver', 'Driver distance from crossing', driverDistance, limits.minimumDriverDistance, limits.captureDistance, onDriver, Eye)}
    </div>
  </div>;
}
