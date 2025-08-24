const http = require('http');
const fs = require('fs');
const path = require('path');
const { URLSearchParams } = require('url');

const envVars = [
  'WEBHOOK_URL',
  'MINIO_ROOT_USER',
  'MINIO_ROOT_PASSWORD',
  'BASEROW_PUBLIC_URL',
  'BASEROW_API_TOKEN',
  'API_KEY',
  'S3_ENDPOINT_URL',
  'S3_ACCESS_KEY',
  'S3_SECRET_KEY',
  'S3_BUCKET_NAME',
  'S3_REGION'
];

function formHTML() {
  return `<html><body><form method="POST">
  ${envVars.map(v => `<label>${v} <input type="text" name="${v}"></label><br>`).join('\n')}
  <button type="submit">Salvar</button>
</form></body></html>`;
}

const server = http.createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/') {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.end(formHTML());
  } else if (req.method === 'POST' && req.url === '/') {
    let body = '';
    req.on('data', chunk => body += chunk.toString());
    req.on('end', () => {
      const params = new URLSearchParams(body);
      const content = envVars.map(v => `${v}=${params.get(v) || ''}`).join('\n');
      const envPath = path.join(__dirname, '..', '.env');
      fs.writeFileSync(envPath, content);
      res.writeHead(200, {'Content-Type': 'text/plain'});
      res.end('Variáveis salvas em .env');
    });
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Servidor webfront rodando na porta ${PORT}`);
});
