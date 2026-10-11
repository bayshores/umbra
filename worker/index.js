import { handleNemotron } from './nemotron.js';

// Hosted Nemotron stays off whatever the host variables say: an in-memory budget restarts at $0 in every isolate, so it cannot
// bound total spending. Turning it on needs an authorized hosted run and a durable shared budget first. Until then the route
// still applies its request checks and answers 503 assistant_unavailable without contacting upstream.
export default {
  async fetch(request, env) {
    if (new URL(request.url).pathname === '/api/nemotron') return handleNemotron(request, {}, {});
    return env.ASSETS.fetch(request);
  },
};
