import { useEffect, useState } from 'react';

// Shared switches for Sightline's motion: every animated feature stops while the tab is hidden and stays off when the
// computer's Reduce Motion setting is on. Both hooks follow changes made while the page is open.
const QUERY = '(prefers-reduced-motion: reduce)';
export const prefersReducedMotion = () => typeof matchMedia === 'function' && matchMedia(QUERY).matches;

export function useReducedMotion() {
  const [reduced, setReduced] = useState(prefersReducedMotion);
  useEffect(() => {
    if (typeof matchMedia !== 'function') return undefined;
    const query = matchMedia(QUERY), change = () => setReduced(query.matches);
    change();
    query.addEventListener('change', change);
    return () => query.removeEventListener('change', change);
  }, []);
  return reduced;
}

export function usePageVisible() {
  const [visible, setVisible] = useState(() => typeof document === 'undefined' || !document.hidden);
  useEffect(() => {
    const change = () => setVisible(!document.hidden);
    change();
    document.addEventListener('visibilitychange', change);
    return () => document.removeEventListener('visibilitychange', change);
  }, []);
  return visible;
}
