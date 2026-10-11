/**
 * TrustGuard-FL API Client.
 *
 * Communicates with the FastAPI central coordinator service.
 * Supports configurable API base URL via VITE_API_BASE_URL or localStorage override.
 */

export const getApiBase = (): string => {
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem('trustguard_api_base');
    if (stored) return stored;
  }
  const envBase = (import.meta as any).env?.VITE_API_BASE_URL;
  return envBase || '/api';
};

export const setApiBase = (url: string) => {
  if (typeof window !== 'undefined') {
    localStorage.setItem('trustguard_api_base', url);
  }
};

const handleResponse = async (res: Response) => {
  if (!res.ok) {
    let detail = `HTTP Error ${res.status}: ${res.statusText}`;
    try {
      const err = await res.json();
      detail = err.detail || err.message || detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json();
};

export const fetchStatus = async () => {
  const res = await fetch(`${getApiBase()}/status`);
  return handleResponse(res);
};

export const fetchClients = async () => {
  const res = await fetch(`${getApiBase()}/clients`);
  return handleResponse(res);
};

export const fetchRounds = async () => {
  const res = await fetch(`${getApiBase()}/fl/rounds`);
  return handleResponse(res);
};

export const fetchActiveRound = async () => {
  const res = await fetch(`${getApiBase()}/fl/rounds/active`);
  return handleResponse(res);
};

export const startNewRound = async (targetClients: number = 2) => {
  const res = await fetch(`${getApiBase()}/fl/rounds/start?target_clients=${targetClients}`, {
    method: 'POST',
  });
  return handleResponse(res);
};

export const fetchEvents = async (params: { client_id?: string; severity?: string; layer?: string; limit?: number; offset?: number } = {}) => {
  const query = new URLSearchParams();
  if (params.client_id) query.set('client_id', params.client_id);
  if (params.severity) query.set('severity', params.severity);
  if (params.layer) query.set('layer', params.layer);
  if (params.limit) query.set('limit', String(params.limit));
  if (params.offset) query.set('offset', String(params.offset));

  const qs = query.toString();
  const url = `${getApiBase()}/security/events${qs ? `?${qs}` : ''}`;
  const res = await fetch(url);
  return handleResponse(res);
};

export const fetchHealth = async () => {
  const res = await fetch(`${getApiBase()}/system/health`);
  return handleResponse(res);
};

export const demoAes = async (payload: string) => {
  const res = await fetch(`${getApiBase()}/demo/aes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ payload }),
  });
  return handleResponse(res);
};

export const demoHmac = async (payload: string) => {
  const res = await fetch(`${getApiBase()}/demo/hmac`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ payload }),
  });
  return handleResponse(res);
};

export const demoIsolationForest = async () => {
  const res = await fetch(`${getApiBase()}/demo/isolation-forest`, { method: 'POST' });
  return handleResponse(res);
};

export const predictCsv = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${getApiBase()}/predict`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse(res);
};
