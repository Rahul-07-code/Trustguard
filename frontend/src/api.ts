const API_BASE = 'http://localhost:8000/api';

export const fetchStatus = async () => (await fetch(`${API_BASE}/status`)).json();
export const fetchClients = async () => (await fetch(`${API_BASE}/clients`)).json();
export const fetchRounds = async () => (await fetch(`${API_BASE}/fl/rounds`)).json();
export const fetchEvents = async () => (await fetch(`${API_BASE}/security/events`)).json();
export const fetchHealth = async () => (await fetch(`${API_BASE}/system/health`)).json();

export const demoAes = async (payload: string) => {
  const res = await fetch(`${API_BASE}/demo/aes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ payload })
  });
  return res.json();
};

export const demoHmac = async (payload: string) => {
  const res = await fetch(`${API_BASE}/demo/hmac`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ payload })
  });
  return res.json();
};

export const demoIsolationForest = async () => {
  const res = await fetch(`${API_BASE}/demo/isolation-forest`, { method: 'POST' });
  return res.json();
};

export const predictCsv = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Failed to predict');
  }
  return res.json();
};
