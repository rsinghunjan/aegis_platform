import http from 'k6/http';
import { check, sleep } from 'k6';

const API_BASE = (__ENV.API_BASE || 'http://localhost:8000').replace(/\/$/, '');
const TENANT_ID = __ENV.TENANT_ID;
const AUTH_TOKEN = __ENV.AEGIS_AUTH_TOKEN;

if (!TENANT_ID || !AUTH_TOKEN) {
  throw new Error('TENANT_ID and AEGIS_AUTH_TOKEN must be configured');
}

export const options = {
  stages: [
    { duration: '30s', target: 5 },
    { duration: '60s', target: 25 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<2000'],
  },
};

export default function () {
  const response = http.post(
    `${API_BASE}/ai/answer`,
    JSON.stringify({
      tenant_id: TENANT_ID,
      query: 'Summarize the configured test knowledge.',
    }),
    {
      headers: {
        Authorization: 'Bearer ' + AUTH_TOKEN,
        'Content-Type': 'application/json',
      },
    },
  );

  check(response, {
    'AI answer succeeds': (result) => result.status === 200,
    'AI answer includes citations': (result) =>
      result.status === 200 && Array.isArray(result.json('citations')),
  });
  sleep(0.25);
}
