import { useEffect, useRef } from 'react';
import { PersonSimple, Eye } from '@phosphor-icons/react';
import * as THREE from 'three';
import { planCamera } from './plan-camera.mjs';
import { driverViewport, comparisonScissors, calibrationCamera } from './driver-viewport.mjs';
import { renderGeometry } from './scene.mjs';
import { mapLayers, rulerDots, rulerPoint, setbackAt } from './map-layers.mjs';

// Person marker colour per result: only Blocked carries the accent; Unknown stays neutral.
const STATES = { clear: 0xf4f2ff, blocked: 0xff7a9a, unknown: 0xf4f2ff };
const VIEW_BACKGROUND = 0x060716;
// Dot diameter in metres before the per-pixel clamp: near points read larger, far points smaller.
const POINT_SIZE_M = 0.045;
// Each dot's stored colour is the recorded camera's (grayscale) brightness at its pixel. Dots are shaded by that
// brightness and given a quiet grey ramp by distance from the driver (top-down view: by height above the ground).
const RAMP = { near: new THREE.Color(0x9ea3b0), mid: new THREE.Color(0x7f8393), far: new THREE.Color(0x5f6271) };
const UP = new THREE.Vector3(0, 1, 0);
// Night-map palette (top-down view): road surface, curbs, lane paint, crosswalk, annotated context objects, ruler dots.
const MAP = { road: 0x141a2c, curb: 0x8a90b8, yellow: 0xcdb36a, white: 0xeceaf6, crossing: 0xe9e4f2, context: 0xb9b6cc, cyclist: 0xa9c7e8 };
const RULER = { blocked: new THREE.Color(0xff7a9a), unknown: new THREE.Color(0xe9e4f2), overlap: new THREE.Color(0xa9c7e8), none: new THREE.Color(0x646a8c) };
const TICKS = [0, 5, 10, 15];
// Vehicles this long or longer are drawn as a cab and a cargo box; shorter ones as a car.
const TRUCK_LENGTH_M = 6;
const metres = n => Number(n).toFixed(2);
const finite3 = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const flat = v => { const n = Math.hypot(v[0], v[2]); return [v[0] / n, 0, v[2] / n]; };
const overlaps = (r, others, pad = 3) => others.some(o => o && r.left < o.right + pad && r.right > o.left - pad && r.top < o.bottom + pad && r.bottom > o.top - pad);

function validate(scene, eye, scenario) {
  const c = scene?.camera;
  const surface = scene ? renderGeometry(scene) : null;
  if (!surface || !(surface.positions instanceof Float32Array) || !(surface.colors instanceof Float32Array) ||
      !(surface.triangles instanceof Uint32Array) || surface.positions.length < 3 || surface.positions.length % 3 ||
      surface.colors.length !== surface.positions.length || surface.triangles.length % 3 ||
      !finite3(surface.bounds?.min) || !finite3(surface.bounds?.max)) return 'Reconstruction rendering data is incomplete or invalid.';
  if (!(scene?.positions instanceof Float32Array) || scene.positions.length < 3 || scene.positions.length % 3 ||
      !(scene.colors instanceof Float32Array) || scene.colors.length !== scene.positions.length ||
      !(scene.triangles instanceof Uint32Array) || scene.triangles.length % 3 ||
      !finite3(scene.bounds?.min) || !finite3(scene.bounds?.max) || !finite3(c?.eye) ||
      !finite3(c?.forward) || !finite3(c?.up) ||
      ![c?.fx, c?.fy, c?.cx, c?.cy, c?.width, c?.height].every(Number.isFinite) ||
      c.fx <= 0 || c.fy <= 0 || c.width <= 0 || c.height <= 0) return 'Measured scene data is incomplete or invalid.';
  if (scene.bounds.min.some((n, i) => n > scene.bounds.max[i]) ||
      Math.hypot(...c.forward) === 0 || Math.hypot(...c.up) === 0 ||
      new THREE.Vector3(...c.forward).normalize().cross(new THREE.Vector3(...c.up).normalize()).length() < 1e-6)
    return 'Measured scene bounds or camera orientation is invalid.';
  if (eye != null && !finite3(eye)) return 'Camera eye must contain three finite coordinates.';
  for (const n of surface.positions) if (!Number.isFinite(n)) return 'Scene reconstruction contains non-finite point data.';
  for (const n of surface.colors) if (!Number.isFinite(n) || n < 0 || n > 1) return 'Scene reconstruction contains invalid RGB color data.';
  for (const i of surface.triangles) if (i >= surface.positions.length / 3) return 'Scene reconstruction contains an invalid triangle index.';
  if (scenario) {
    const points = [scenario.target, scenario.targetGround, scenario.baseline?.center, scenario.proposed?.center, scenario.axis, scenario.crossing, scenario.baseline?.axis, scenario.proposed?.axis];
    const dims = [scenario.vehicle, scenario.baseline?.vehicle, scenario.proposed?.vehicle].flatMap(v => [v?.width, v?.height, v?.length]);
    const states = [scenario.currentVisibility?.state, scenario.proposedVisibility?.state];
    if (!points.every(finite3) || !dims.every(n => Number.isFinite(n) && n > 0) ||
        ![scenario.baselineSetback, scenario.proposedSetback].every(Number.isFinite) ||
        !states.every(s => Object.hasOwn(STATES, s)) || [scenario.axis, scenario.baseline.axis, scenario.proposed.axis].some(a => Math.hypot(a[0], a[2]) === 0))
      return 'Comparison geometry or visibility state is invalid.';
  }
  return null;
}

function disposeTree(root) {
  root.traverse(o => {
    o.geometry?.dispose();
    (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m?.dispose());
  });
}
function addMarker(point, color, group) {
  const marker = new THREE.Mesh(new THREE.SphereGeometry(0.08, 14, 10), new THREE.MeshBasicMaterial({ color }));
  marker.position.set(...point);
  group.add(marker);
}
// A hidden ground point (behind an annotated object in the recorded image) is a hollow ring facing the viewer.
function addHollowMarker(point, color, group, facing) {
  const ring = Array.from({ length: 24 }, (_, i) => new THREE.Vector3(Math.cos(i / 24 * Math.PI * 2) * 0.12, Math.sin(i / 24 * Math.PI * 2) * 0.12, 0));
  const marker = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(ring), new THREE.LineBasicMaterial({ color }));
  marker.position.set(...point);
  marker.lookAt(...facing);
  group.add(marker);
}
function addTarget(point, ground, color, group, hiddenGround = false, facing = null) {
  if (!finite3(point) || !finite3(ground)) throw new Error('Measured target ground point is missing or invalid.');
  addMarker(point, color, group);
  const stem = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...ground), new THREE.Vector3(...point)]);
  group.add(new THREE.Line(stem, new THREE.LineBasicMaterial({ color })));
  if (hiddenGround) addHollowMarker(ground, color, group, facing ?? [ground[0], ground[1] + 1, ground[2]]);
  else addMarker(ground, color, group);
}
// The LiDAR points as soft round dots. Driver views size each dot by its distance (pixelsPerMetre / depth);
// the top-down view uses one fixed size. Points never write depth, so opaque vehicles hide the points behind them.
function pointMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { pixelsPerMetre: { value: 1 }, fixedSize: { value: 0 }, groundY: { value: 0 },
      nearColor: { value: RAMP.near }, midColor: { value: RAMP.mid }, farColor: { value: RAMP.far } },
    vertexShader: `
      uniform float pixelsPerMetre;
      uniform float fixedSize;
      uniform float groundY;
      uniform vec3 nearColor;
      uniform vec3 midColor;
      uniform vec3 farColor;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vec4 view = modelViewMatrix * vec4(position, 1.0);
        float depth = max(-view.z, 0.1);
        float t;
        if (fixedSize > 0.0) {
          t = 1.0 - clamp((position.y - groundY) / 6.0, 0.0, 1.0);
          gl_PointSize = fixedSize;
        } else {
          t = clamp((depth - 4.0) / 26.0, 0.0, 1.0);
          gl_PointSize = clamp(pixelsPerMetre / depth, 1.0, 7.0);
        }
        vec3 ramp = t < 0.5 ? mix(nearColor, midColor, t * 2.0) : mix(midColor, farColor, t * 2.0 - 1.0);
        vColor = ramp * (0.5 + 0.9 * sqrt(dot(color, vec3(0.3333))));
        vAlpha = fixedSize > 0.0 ? 0.9 : mix(1.0, 0.6, t);
        gl_Position = projectionMatrix * view;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        float r = length(gl_PointCoord - 0.5);
        if (r > 0.5) discard;
        gl_FragColor = vec4(min(vColor, vec3(1.0)), smoothstep(0.5, 0.05, r) * vAlpha);
        #include <colorspace_fragment>
      }`,
    vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
}
// Vehicles are opaque near-black volumes with bright edges.
function addVehicle(center, dimensions, axis, group) {
  const geometry = new THREE.BoxGeometry(dimensions.width, dimensions.height, dimensions.length);
  const yaw = Math.atan2(axis[0], axis[2]);
  const fill = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: VIEW_BACKGROUND, side: THREE.DoubleSide }));
  fill.position.set(...center);
  fill.rotation.y = yaw;
  group.add(fill);
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color: 0xf4f2ff }));
  edge.position.copy(fill.position);
  edge.rotation.copy(fill.rotation);
  group.add(edge);
  return fill;
}

function flatMaterial(color, opacity = 1) {
  return new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthTest: false, depthWrite: false, side: THREE.DoubleSide });
}
function lineMaterial(color, opacity = 1) {
  return new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, depthTest: false, depthWrite: false });
}
// A flat ribbon along a polyline (lane paint), `offset` metres to its side, on the ground.
function ribbon(points, halfWidth, offset, material, dashed = false) {
  const positions = [], index = [];
  const cut = dashed ? (d => d % 12 < 3) : (() => true);
  let travelled = 0;
  for (let i = 0; i + 1 < points.length; i++) {
    const a = points[i], b = points[i + 1], dx = b[0] - a[0], dz = b[2] - a[2], len = Math.hypot(dx, dz);
    if (!(len > 0)) continue;
    const nx = -dz / len, nz = dx / len, steps = Math.max(1, Math.ceil(len / 1.5));
    for (let k = 0; k < steps; k++) {
      const t0 = k / steps, t1 = (k + 1) / steps, mid = travelled + len * (t0 + t1) / 2;
      if (!cut(mid)) continue;
      const p = t => [a[0] + dx * t, a[1] + (b[1] - a[1]) * t + .02, a[2] + dz * t], q0 = p(t0), q1 = p(t1), base = positions.length / 3;
      for (const [q, side] of [[q0, -1], [q0, 1], [q1, -1], [q1, 1]]) positions.push(q[0] + nx * (offset + side * halfWidth), q[1], q[2] + nz * (offset + side * halfWidth));
      index.push(base, base + 1, base + 2, base + 1, base + 3, base + 2);
    }
    travelled += len;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(index);
  return new THREE.Mesh(geometry, material);
}
// The publisher map and annotated objects, drawn flat under the scan points (mapped-ground scenes only).
function buildMapBase(layers) {
  const group = new THREE.Group(), y = layers.crossing.reduce((sum, p) => sum + p[1], 0) / layers.crossing.length - .05;
  const order = (object, n) => { object.renderOrder = n; group.add(object); };
  for (const outline of layers.areas) {
    const contour = outline.map(([x, z]) => new THREE.Vector2(x, z));
    const faces = THREE.ShapeUtils.triangulateShape(contour, []);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(outline.flatMap(([x, z]) => [x, y, z]), 3));
    geometry.setIndex(faces.flat());
    order(new THREE.Mesh(geometry, flatMaterial(MAP.road)), -20);
    order(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(outline.map(([x, z]) => new THREE.Vector3(x, y + .01, z))), lineMaterial(MAP.curb, .28)), -19);
  }
  const band = new THREE.BufferGeometry();
  band.setAttribute('position', new THREE.Float32BufferAttribute(layers.crossing.flatMap(p => [p[0], p[1] + .03, p[2]]), 3));
  band.setIndex([0, 1, 2, 0, 2, 3]);
  order(new THREE.Mesh(band, flatMaterial(MAP.crossing, .13)), -18);
  order(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(layers.crossing.map(p => new THREE.Vector3(p[0], p[1] + .04, p[2]))), lineMaterial(MAP.crossing, .6)), -17);
  for (const { line, style } of layers.lanes) {
    const material = flatMaterial(style.color === 'yellow' ? MAP.yellow : MAP.white, .95);
    for (const offset of style.double ? [-.11, .11] : [0]) order(ribbon(line, style.double ? .05 : .065, offset, material, style.dashed), -16);
  }
  const dots = [];
  for (const o of layers.objects) {
    if (o.kind === 'person' || o.kind === 'other') { dots.push(o); continue; }
    const loop = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(o.corners.map(([x, z]) => new THREE.Vector3(x, o.ground + .05, z))),
      lineMaterial(o.kind === 'cyclist' ? MAP.cyclist : MAP.context, o.kind === 'cyclist' ? .55 : .32));
    loop.userData.center = o.center;
    order(loop, -15);
  }
  if (dots.length) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(dots.flatMap(o => [o.center[0], o.ground + .05, o.center[2]]), 3));
    order(new THREE.Points(geometry, new THREE.PointsMaterial({ color: MAP.context, size: 4, sizeAttenuation: false, transparent: true, opacity: .55, depthTest: false })), -14);
  }
  return group;
}
// Round ruler dots of a fixed pixel size, coloured by the checked distance state at each setback.
function rulerObject(dots, pixelRatio) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(dots.flatMap(d => [d.position[0], d.position[1] + .06, d.position[2]]), 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(dots.flatMap(d => (RULER[d.state] ?? RULER.none).toArray()), 3));
  const material = new THREE.ShaderMaterial({
    uniforms: { size: { value: 4.5 * pixelRatio } },
    vertexShader: 'uniform float size; varying vec3 vColor; void main(){ vColor = color; gl_PointSize = size; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'varying vec3 vColor; void main(){ float r = length(gl_PointCoord - .5); if (r > .5) discard; gl_FragColor = vec4(vColor, smoothstep(.5, .32, r)); }',
    vertexColors: true, transparent: true, depthTest: false, depthWrite: false,
  });
  const points = new THREE.Points(geometry, material);
  points.renderOrder = -10;
  return points;
}

export default function StreetView({ scene, scenario, preview, eye, mode = 'driver', split = .5, baselineLabel = 'Current vehicle', framing = 'tight', ruler = null, onSetbackChange, onDriverChange, onError }) {
  const host = useRef(null), api = useRef(null), labels = useRef({}), latest = useRef(null), pending = useRef({});
  latest.current = { scene, scenario, preview, eye, mode, split, framing, onSetbackChange, onDriverChange, onError };

  useEffect(() => {
    const element = host.current, invalid = validate(scene, eye, scenario);
    if (!element) return undefined;
    if (invalid) { onError?.(invalid); return undefined; }
    const world = new THREE.Scene();
    world.background = mode === 'recorded' ? null : new THREE.Color(VIEW_BACKGROUND);
    let renderer, observer, removePointer, removeResolution, dots = null, mapBase = null;
    const contextLost = () => latest.current.onError?.('Measured scene rendering failed: WebGL context lost.');
    try {
      // Transparent outside the calibrated image: a driver view letterboxed in a wider stage shows the page behind it.
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setClearColor(0x000000, 0);
      renderer.domElement.addEventListener('webglcontextlost', contextLost);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.domElement.setAttribute('aria-hidden', 'true');
      renderer.domElement.style.cssText = 'display:block;width:100%;height:100%;touch-action:none;';
      element.appendChild(renderer.domElement);
      if (mode !== 'recorded') {
        const surface = renderGeometry(scene);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(surface.positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(surface.colors, 3));
        if (surface.triangles.length) {
          geometry.setIndex(new THREE.BufferAttribute(surface.triangles, 1));
          world.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })));
        }
        const pointGeometry = new THREE.BufferGeometry();
        pointGeometry.setAttribute('position', geometry.getAttribute('position'));
        pointGeometry.setAttribute('color', geometry.getAttribute('color'));
        dots = pointMaterial();
        world.add(new THREE.Points(pointGeometry, dots));
        // The night map: real publisher map layers and annotated objects under the scan points.
        const layers = mode === 'plan' ? mapLayers(scene.ground) : null;
        if (layers) world.add(mapBase = buildMapBase(layers));
      }
      const draw = () => {
        const config = latest.current, width = element.clientWidth, height = element.clientHeight;
        if (!width || !height) return null;
        if (renderer.getPixelRatio() !== window.devicePixelRatio) renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(width, height, false);
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, width, height);
        renderer.clear(true, true, true);
        let camera, viewport;
        if (config.mode === 'plan') {
          camera = planCamera(scene, config.scenario, width / height, config.framing === 'context');
          if (dots) {
            dots.uniforms.fixedSize.value = 2 * renderer.getPixelRatio();
            // Mapped-ground scenes keep the ground at the vehicle's ground height; measured-depth scenes at Y 0.
            dots.uniforms.groundY.value = config.scenario?.vehicleOrigin ? config.scenario.vehicleGroundHeight : 0;
          }
        } else {
          const viewEye = config.eye ?? scene.camera.eye;
          let far = 0;
          const surfaceBounds = renderGeometry(scene).bounds;
          const bounds = {min:scene.bounds.min.map((n,i)=>Math.min(n,surfaceBounds.min[i])),
            max:scene.bounds.max.map((n,i)=>Math.max(n,surfaceBounds.max[i]))};
          for (const x of bounds.min[0] === bounds.max[0] ? [bounds.min[0]] : [bounds.min[0], bounds.max[0]])
            for (const y of [bounds.min[1], bounds.max[1]]) for (const z of [bounds.min[2], bounds.max[2]])
              far = Math.max(far, new THREE.Vector3(x, y, z).distanceTo(new THREE.Vector3(...viewEye)));
          if (!(far > 0)) throw new Error('Measured scene bounds do not surround the selected camera.');
          camera = calibrationCamera(scene.camera, viewEye, far * 2);
          viewport = driverViewport(width, height, scene.camera);
          // Device pixels per metre at 1 m depth for this calibrated view (focal length scaled to the drawn viewport).
          if (dots) dots.uniforms.pixelsPerMetre.value = POINT_SIZE_M * scene.camera.fy * (viewport.height / scene.camera.height) * renderer.getPixelRatio();
          renderer.setViewport(viewport.left, viewport.bottom, viewport.width, viewport.height);
          renderer.setScissor(viewport.left, viewport.bottom, viewport.width, viewport.height);
          renderer.setScissorTest(true);
        }
        if (config.mode === 'compare' && api.current?.currentOverlay) {
          // Three.js r186 comparison example: one camera/viewport, two scissor regions.
          // Keep calibrated projection unchanged as the split moves.
          const regions = comparisonScissors(viewport, width, config.split);
          for (let i = 0; i < 2; i++) {
            api.current.currentOverlay.visible = i === 0;
            api.current.proposedOverlay.visible = i === 1;
            if (regions[i][2] > 0) {
              renderer.setScissor(...regions[i]);
              renderer.render(world, camera);
            }
          }
        } else renderer.render(world, camera);
        renderer.setScissorTest(false);
        return camera;
      };
      api.current = { world, renderer, draw, mapBase, plane: new THREE.Plane(UP, 0), ray: new THREE.Raycaster() };
      const redraw = () => { try { api.current.camera = draw(); placeLabels(); } catch (error) { latest.current.onError?.(error.message); } };
      const watchResolution = () => {
        removeResolution?.();
        const query = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
        const changed = () => { watchResolution(); redraw(); };
        query.addEventListener('change', changed);
        removeResolution = () => query.removeEventListener('change', changed);
      };
      observer = new ResizeObserver(redraw);
      observer.observe(element);
      watchResolution();
      for (const [type, handler] of [['pointerdown', pointerDown], ['pointermove', pointerMove], ['pointerup', pointerUp], ['pointercancel', pointerUp]]) element.addEventListener(type, handler);
      removePointer = () => {
        for (const [type, handler] of [['pointerdown', pointerDown], ['pointermove', pointerMove], ['pointerup', pointerUp], ['pointercancel', pointerUp]]) element.removeEventListener(type, handler);
      };
      api.current.camera = draw();
    } catch (error) {
      observer?.disconnect();
      removeResolution?.();
      removePointer?.();
      renderer?.domElement.removeEventListener('webglcontextlost', contextLost);
      renderer?.dispose();
      element.replaceChildren();
      api.current = null;
      latest.current.onError?.(`Measured scene rendering failed: ${error.message}`);
    }
    return () => {
      observer?.disconnect();
      removeResolution?.();
      removePointer?.();
      renderer?.domElement.removeEventListener('webglcontextlost', contextLost);
      if (api.current?.overlay) disposeTree(api.current.overlay);
      disposeTree(world);
      renderer?.dispose();
      if (renderer?.domElement.parentNode === element) element.removeChild(renderer.domElement);
      api.current = null;
    };
  }, [scene, mode]);

  function pointerDown(event) {
    const a = api.current, config = latest.current, vehicle = event.target.closest?.('.map-vehicle--proposed');
    if (!a?.camera || !vehicle || config.mode !== 'plan' || !config.onSetbackChange || event.button !== 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    a.ray.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -((event.clientY - rect.top) / rect.height * 2 - 1)), a.camera);
    // Mapped-ground scenes have the camera at the origin, so drag on the vehicle's ground height rather than Y 0.
    a.plane.constant = config.scenario.vehicleOrigin ? -config.scenario.vehicleGroundHeight : 0;
    const point = a.ray.ray.intersectPlane(a.plane, new THREE.Vector3());
    if (!point) return;
    // The what-if moves along the lane axis: the setback changes by the pointer's movement along it from the grab point.
    const s = config.scenario, axis = new THREE.Vector3(s.axis[0], 0, s.axis[2]).normalize();
    a.grab = { along: point.dot(axis), setback: s.proposedSetback, vehicle };
    a.dragging = event.pointerId;
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    vehicle.focus({ focusVisible: false });
    vehicle.classList.add('is-dragging');
  }
  function pointerMove(event) {
    const a = api.current, config = latest.current;
    if (a?.dragging == null || a.dragging !== event.pointerId) return;
    const rect = event.currentTarget.getBoundingClientRect();
    a.ray.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -((event.clientY - rect.top) / rect.height * 2 - 1)), a.camera);
    a.plane.constant = config.scenario.vehicleOrigin ? -config.scenario.vehicleGroundHeight : 0;
    const point = a.ray.ray.intersectPlane(a.plane, new THREE.Vector3());
    if (!point) return;
    const s = config.scenario, axis = new THREE.Vector3(s.axis[0], 0, s.axis[2]).normalize();
    const setback = a.grab.setback + point.dot(axis) - a.grab.along;
    // Dragging the box stops at the slider's ends, as the ruler handle does: its nearest point (a turned annotated pose: its
    // nearer front corner) keeps the road tolerance from the crossing, and its rear stays inside the captured road.
    if (Number.isFinite(setback)) config.onSetbackChange(clampTo(setback, s.limits.minimumSetback, s.limits.maximumSetback));
  }
  function pointerUp(event) {
    const a = api.current;
    if (a?.dragging !== event.pointerId) return;
    a.dragging = null;
    a.grab?.vehicle.classList.remove('is-dragging');
  }
  function placeLabels() {
    const a = api.current, s = latest.current.scenario, nodes = labels.current;
    if (!a?.camera || latest.current.mode !== 'plan' || !s?.whatIf) return;
    const rect = host.current.getBoundingClientRect(), camera = a.camera, perMetre = rect.height / (camera.top - camera.bottom);
    const screen = point => {
      const p = new THREE.Vector3(...point).project(camera);
      return { x: (p.x + 1) * rect.width / 2, y: (1 - p.y) * rect.height / 2, inside: p.z > -1 && p.z < 1 && Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1 };
    };
    const put = (node, at, show = true) => {
      if (!node) return;
      node.style.display = show && at.inside ? '' : 'none';
      node.style.left = `${at.x}px`;
      node.style.top = `${at.y}px`;
    };
    // A vehicle is drawn as its exact footprint (width by length on the ground) at its own heading, nose toward the crossing.
    const footprint = (node, box, show = true) => {
      const along = flat(box.axis), at = screen(box.center), nose = screen(box.center.map((v, i) => v - along[i] * box.vehicle.length / 2));
      const turn = Math.atan2(nose.x - at.x, at.y - nose.y);
      put(node, at, show);
      if (!node) return at;
      node.style.width = `${box.vehicle.width * perMetre}px`;
      node.style.height = `${box.vehicle.length * perMetre}px`;
      node.style.transform = `translate(-50%, -50%) rotate(${turn}rad)`;
      node.style.setProperty('--corner', `${Math.min(box.vehicle.length >= TRUCK_LENGTH_M ? .28 : .42, box.vehicle.width * .3) * perMetre}px`);
      // The shadow falls toward the bottom of the screen at any heading (box-shadow offsets turn with the box).
      node.style.setProperty('--shadow-x', `${(3 * Math.sin(turn)).toFixed(2)}px`);
      node.style.setProperty('--shadow-y', `${(3 * Math.cos(turn)).toFixed(2)}px`);
      return at;
    };
    const b = s.baseline, p = s.proposed;
    const sameFootprint = Math.hypot(b.center[0] - p.center[0], b.center[2] - p.center[2]) < .05 && ['width', 'length'].every(k => Math.abs(b.vehicle[k] - p.vehicle[k]) < .01);
    footprint(nodes.baseline, b, !sameFootprint);
    const at = footprint(nodes.setback, p);
    put(nodes.target, screen(s.target));
    put(nodes.driver, screen([s.eye[0], rulerPoint(s, 0)[1], s.eye[2]]));
    // Ruler labels sit just beyond the ruler on the driver's right, reading away from it.
    const zero = screen(rulerPoint(s, 0)), outward = screen(rulerPoint(s, 0, 1)), dx = outward.x - zero.x, dy = outward.y - zero.y, n = Math.hypot(dx, dy) || 1;
    for (const m of TICKS) {
      const node = nodes['tick-' + m], tick = screen(rulerPoint(s, m));
      put(node, { ...tick, x: tick.x + dx / n * 4, y: tick.y + dy / n * 4 }, m <= s.limits.captureDistance);
      if (node) node.style.transform = `translate(${(dx / n - 1) * 50}%, ${(dy / n - 1) * 50}%)`;
    }
    // The distance reads under the proposed vehicle; above it when that would cover the person or the driver, or leave the map.
    // Labels collide by their text, not the soft pool around it (7 by 3 px of padding).
    const value = nodes.value, seen = (node, text = false) => {
      if (!node || node.style.display === 'none') return null;
      const r = node.getBoundingClientRect(), x = text ? 7 : 0, y = text ? 3 : 0;
      return { left: r.left + x, right: r.right - x, top: r.top + y, bottom: r.bottom - y };
    };
    if (value) {
      const along = flat(p.axis), across = [along[2], 0, -along[0]];
      const ys = [-1, 1].flatMap(e => [-1, 1].map(w => screen(p.center.map((v, i) => v + along[i] * e * p.vehicle.length / 2 + across[i] * w * p.vehicle.width / 2)).y));
      put(value, { ...at, y: Math.max(...ys) });
      value.style.transform = 'translate(-50%, 0)';
      const box = seen(value, true), avoid = [seen(nodes.target), seen(nodes.driver)];
      if (box && (overlaps(box, avoid) || box.bottom > rect.bottom - 4)) {
        put(value, { ...at, y: Math.min(...ys) });
        value.style.transform = 'translate(-50%, -100%)';
      }
    }
    // A ruler label gives way to the person, the driver, the distance and the vehicles.
    const keep = [seen(nodes.target), seen(nodes.driver), seen(value, true), seen(nodes.setback), seen(nodes.baseline)];
    for (const m of TICKS) {
      const node = nodes['tick-' + m], box = seen(node, true);
      if (box && overlaps(box, keep)) node.style.display = 'none';
    }
  }
  // Handles drag along the setback axis (the ruler), or step with the arrow keys (Shift: 1 m), within the limits.
  const limitsFor = (kind, s) => kind === 'driver'
    ? [s.driverDistance, s.limits.minimumDriverDistance, s.limits.captureDistance, latest.current.onDriverChange]
    : [s.proposedSetback, s.limits.minimumSetback, s.limits.maximumSetback, latest.current.onSetbackChange];
  const clampTo = (v, min, max) => Math.min(max, Math.max(min, v));
  function handleDrag(kind) {
    return event => {
      const a = api.current, s = latest.current.scenario;
      if (!a?.camera || !s || event.button !== 0) return;
      event.preventDefault();
      const node = event.currentTarget, groundY = rulerPoint(s, 0)[1];
      node.setPointerCapture(event.pointerId);
      node.focus({ focusVisible: false });
      const move = e => {
        const rect = host.current.getBoundingClientRect(), sc = latest.current.scenario;
        a.ray.setFromCamera(new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1, -((e.clientY - rect.top) / rect.height * 2 - 1)), a.camera);
        a.plane.constant = -groundY;
        const hit = a.ray.ray.intersectPlane(a.plane, new THREE.Vector3());
        if (!hit || !sc) return;
        const [, min, max, change] = limitsFor(kind, sc);
        change?.(clampTo(setbackAt(sc, [hit.x, hit.y, hit.z]), min, max));
      };
      const up = () => { node.removeEventListener('pointermove', move); node.removeEventListener('pointerup', up); node.removeEventListener('pointercancel', up); };
      node.addEventListener('pointermove', move); node.addEventListener('pointerup', up); node.addEventListener('pointercancel', up);
    };
  }
  function handleKeys(kind) {
    return event => {
      const s = latest.current.scenario;
      if (!s) return;
      const [value, min, max, change] = limitsFor(kind, s), step = event.shiftKey ? 1 : .1;
      const base = pending.current[kind] ?? value;
      const next = { ArrowLeft: base - step, ArrowDown: base - step, ArrowRight: base + step, ArrowUp: base + step, Home: min, End: max }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      pending.current[kind] = clampTo(next, min, max);
      change?.(pending.current[kind]);
    };
  }
  useEffect(() => { pending.current = {}; }, [scenario]);

  useEffect(() => {
    const a = api.current;
    if (!a) return;
    const previousOverlay = a.overlay;
    try {
      if (previousOverlay) a.world.remove(previousOverlay);
      const group = new THREE.Group();
      a.currentOverlay = null;
      a.proposedOverlay = null;
      // Plan mode looks down; the driver views face the eye.
      const facing = mode === 'plan' ? null : (eye ?? scene.camera.eye);
      if (scenario) {
        const hiddenGround = scenario.support?.target?.hidden === true;
        if (mode === 'compare') {
          a.currentOverlay = new THREE.Group();
          a.proposedOverlay = new THREE.Group();
          addTarget(scenario.target, scenario.targetGround, STATES[scenario.currentVisibility.state], a.currentOverlay, hiddenGround, facing);
          addTarget(scenario.target, scenario.targetGround, STATES[scenario.proposedVisibility.state], a.proposedOverlay, hiddenGround, facing);
          // Each vehicle carries its own pose (an annotated stationary vehicle and its what-if may be turned from the lane axis).
          addVehicle(scenario.baseline.center, scenario.baseline.vehicle, scenario.baseline.axis, a.currentOverlay);
          addVehicle(scenario.proposed.center, scenario.proposed.vehicle, scenario.proposed.axis, a.proposedOverlay);
          group.add(a.currentOverlay, a.proposedOverlay);
        } else if (mode === 'plan') {
          // The top-down view draws its vehicles and the person above the map (placeLabels), so lane paint never crosses them.
          // The annotated box the current vehicle is drawn from is not outlined a second time.
          const own = scenario.baseline.pose === 'annotated' ? scenario.baseline.center : null;
          a.mapBase?.children.forEach(o => { if (o.userData.center) o.visible = !own || Math.hypot(o.userData.center[0] - own[0], o.userData.center[2] - own[2]) > .05; });
          if (scenario.whatIf) group.add(rulerObject(rulerDots(scenario, ruler), a.renderer.getPixelRatio()));
        } else {
          addTarget(scenario.target, scenario.targetGround, STATES[scenario.proposedVisibility.state], group, hiddenGround, facing);
          addVehicle(scenario.proposed.center, scenario.proposed.vehicle, scenario.proposed.axis, group);
        }
      } else if (preview) {
        if (preview.vehicle) addVehicle(preview.vehicle.center, preview.vehicle.dimensions, preview.vehicle.axis, group);
        if (preview.target) addTarget(preview.target, preview.targetGround, STATES.unknown, group, preview.targetHidden === true, facing);
      }
      a.world.add(group);
      a.overlay = group;
      a.camera = a.draw();
      placeLabels();
    } catch (error) { latest.current.onError?.(`Measured scene rendering failed: ${error.message}`); }
    finally {
      // Let replacement materials acquire matching programs before releasing their last previous users.
      if (previousOverlay) disposeTree(previousOverlay);
    }
  }, [scene, scenario, preview, eye, mode, onSetbackChange, ruler]);

  useEffect(() => {
    try { if (api.current) api.current.camera = api.current.draw(); }
    catch (error) { latest.current.onError?.(error.message); }
  }, [split]);

  const sourceLabel = scene.reconstruction ? 'Estimated street reconstruction' : 'Measured street scene';
  const plan = scenario && mode === 'plan' && scenario.whatIf;
  const sliderProps = (kind, label) => {
    const [value, min, max] = limitsFor(kind, scenario);
    return { ref: node => { labels.current[kind] = node; }, role: 'slider', tabIndex: 0, 'aria-label': label,
      'aria-valuemin': Number(metres(min)), 'aria-valuemax': Number(metres(max)), 'aria-valuenow': Number(metres(value)),
      'aria-valuetext': metres(value) + ' metres from the crossing', onKeyDown: handleKeys(kind) };
  };
  const truck = plan && scenario.proposed.vehicle.length >= TRUCK_LENGTH_M;
  const blocked = plan && scenario.proposedVisibility.state === 'blocked';
  const personLabel = 'Person' + (scenario?.support?.target?.supportLabel ? ': ' + scenario.support.target.supportLabel : '');
  return <div ref={host} role={plan ? 'group' : 'img'} aria-label={mode === 'plan' ? `Top-down ${sourceLabel.toLowerCase()}` : mode === 'recorded' ? 'Declared actors over recorded camera view' : `${sourceLabel} from driver view`} style={{ position: 'relative', width: '100%', height: '100%', minHeight: mode === 'recorded' ? 0 : 160, overflow: 'hidden', background: mode === 'plan' ? '#060716' : 'transparent' }}>
    {plan && <>
      <div ref={node => { labels.current.baseline = node; }} className="map-vehicle map-vehicle--baseline" title={baselineLabel}/>
      <div {...sliderProps('setback', 'Proposed vehicle distance from crossing')} className={'map-vehicle map-vehicle--proposed' + (truck ? ' map-vehicle--truck' : '')}>
        {truck ? <><i className="map-vehicle-cab"><i className="map-vehicle-glass"/></i><i className="map-vehicle-cargo"/></>
          : <><i className="map-vehicle-glass"/><i className="map-vehicle-glass map-vehicle-glass--rear"/></>}
      </div>
      <span ref={node => { labels.current.value = node; }} className="map-vehicle-value" aria-hidden="true">{metres(scenario.proposedSetback)} m</span>
      <div ref={node => { labels.current.target = node; }} className={'map-person' + (blocked ? ' map-person--blocked' : '')} role="img" aria-label={personLabel} title={personLabel}><PersonSimple size={14} weight="bold" aria-hidden="true"/></div>
      {TICKS.filter(m => m <= scenario.limits.captureDistance).map(m => <span key={m} ref={node => { labels.current['tick-' + m] = node; }} className="map-tick" aria-hidden="true">{m} m</span>)}
      <div {...sliderProps('driver', 'Driver distance from crossing')} className="map-handle" onPointerDown={handleDrag('driver')}>
        <span className="map-handle-chip"><Eye size={14} weight="bold" aria-hidden="true"/></span><span className="map-handle-value">{metres(scenario.driverDistance)} m</span>
      </div>
    </>}
  </div>;
}
