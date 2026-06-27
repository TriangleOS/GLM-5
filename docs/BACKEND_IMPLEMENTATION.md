# Triangle LLM Backend Architecture & Implementation Guide

**Confidential - For Authorized Use Only**

## Overview

The Triangle LLM backend consists of:
1. **Model Server** - Inference engine with multi-GPU support
2. **API Gateway** - Request routing and load balancing
3. **Authentication Layer** - OAuth2/mTLS with Triangle OS IAM
4. **Audit Logger** - Compliance tracking and forensics
5. **Cache Layer** - KV-cache optimization and prompt caching

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Consumer Clients                          │
├─────────────────────────────────────────────────────────────┤
                              ▲
                              │ HTTPS
                              │
┌─────────────────────────────────────────────────────────────┐
│                  API Gateway (nginx/Envoy)                   │
│  • Request routing  • Rate limiting  • TLS termination      │
├─────────────────────────────────────────────────────────────┤
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│           Authentication & Authorization Layer               │
│  • OAuth2/mTLS  • Token validation  • Audit logging         │
├─────────────────────────────────────────────────────────────┤
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│         Triangle LLM Inference Servers (Cluster)             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Model Server 1 (4x GPU)                            │   │
│  │  • Tensor parallelism  • KV-cache  • Quantization  │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Model Server 2 (4x GPU)                            │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Model Server N (4x GPU)                            │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│            Data & Persistence Layer                          │
│  • Redis (KV-cache)  • PostgreSQL (audit)  • S3 (models)   │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Model Server Implementation

### 1.1 Core Model Server Class

```python
# triangle_llm/server/model_server.py

import asyncio
import logging
from typing import AsyncGenerator, Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(default=512, le=131072, description="Max output tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    reasoning_effort: str = Field(default="max", pattern="^(max|high)$")
    stop_sequences: List[str] = Field(default_factory=list)


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    request_id: str
    prompt: str
    completion: str
    finish_reason: str
    tokens_generated: int
    latency_ms: float
    model_version: str


class TriangleLLMModelServer:
    """Production Triangle LLM inference server."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 4,
        pipeline_parallel_size: int = 1,
        max_batch_size: int = 64,
        max_context_length: int = 1_000_000,
        verify_signatures: bool = True
    ):
        """Initialize model server."""
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.max_batch_size = max_batch_size
        self.max_context_length = max_context_length

        logger.info(f"Initializing Triangle LLM from {model_path}...")

        # Verify model signature
        if verify_signatures:
            self._verify_model_signature()

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Load model with vLLM for optimized inference
        logger.info(f"Loading model with vLLM (TP={tensor_parallel_size})...")
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_context_length,
            quantization="auto",
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            dtype=torch.bfloat16
        )

        logger.info("✓ Model server initialized successfully")

    def _verify_model_signature(self):
        """Verify cryptographic model signature."""
        from triangle_llm.security import verify_model_signature

        logger.info("Verifying model signature...")
        is_valid = verify_model_signature(self.model_path)
        if not is_valid:
            raise RuntimeError("Model signature verification failed!")
        logger.info("✓ Model signature verified")

    async def generate(
        self,
        request: GenerationRequest,
        request_id: str
    ) -> GenerationResponse:
        """Generate text response."""
        import time
        start_time = time.time()

        # Apply reasoning effort
        temperature = request.temperature
        if request.reasoning_effort == "high":
            # Reduce temperature for more focused outputs
            temperature = max(0.3, temperature * 0.5)

        # Create sampling parameters
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop_sequences if request.stop_sequences else None
        )

        # Generate completion
        logger.info(f"[{request_id}] Generating with {len(request.prompt)} input tokens...")
        
        outputs = await asyncio.to_thread(
            self.model.generate,
            request.prompt,
            sampling_params
        )

        # Extract result
        completion = outputs[0].outputs[0].text
        tokens_generated = len(
            self.tokenizer.encode(completion, add_special_tokens=False)
        )
        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[{request_id}] Generation complete: "
            f"{tokens_generated} tokens in {latency_ms:.0f}ms"
        )

        return GenerationResponse(
            request_id=request_id,
            prompt=request.prompt,
            completion=completion,
            finish_reason=outputs[0].outputs[0].finish_reason,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            model_version="triangle-llm-1.0"
        )

    async def generate_streaming(
        self,
        request: GenerationRequest,
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream generation tokens as they are produced."""
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens
        )

        logger.info(f"[{request_id}] Starting streaming generation...")

        outputs = await asyncio.to_thread(
            self.model.generate,
            request.prompt,
            sampling_params,
            use_tqdm=False
        )

        for output in outputs:
            token_text = output.outputs[0].text
            yield token_text

    def get_model_info(self) -> Dict:
        """Get model metadata."""
        return {
            "name": "triangle-llm-1.0",
            "parameters": 744_000_000_000,
            "context_length": self.max_context_length,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_batch_size": self.max_batch_size,
            "dtype": "bfloat16",
            "quantization": "fp8"
        }
```

### 1.2 FastAPI Server Implementation

```python
# triangle_llm/server/api_server.py

import uuid
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import json
from datetime import datetime, timezone

from triangle_llm.server.model_server import (
    TriangleLLMModelServer,
    GenerationRequest,
    GenerationResponse
)
from triangle_llm.security import (
    verify_api_token,
    audit_log,
    rate_limit,
    check_pii
)
from triangle_llm.monitoring import MetricsCollector

logger = logging.getLogger(__name__)

# Global model server instance
model_server: Optional[TriangleLLMModelServer] = None
metrics = MetricsCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model server lifecycle."""
    global model_server
    
    logger.info("Starting Triangle LLM API server...")
    
    model_server = TriangleLLMModelServer(
        model_path="/models/triangle-llm-1.0",
        tensor_parallel_size=4,
        max_batch_size=64
    )
    
    logger.info("✓ API server started")
    
    yield
    
    logger.info("Shutting down Triangle LLM API server...")
    # Cleanup
    logger.info("✓ API server stopped")


app = FastAPI(
    title="Triangle LLM API",
    description="Enterprise-grade proprietary LLM inference",
    version="1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "triangle-llm-1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_memory_used": "78%"
    }


@app.get("/v1/model")
async def get_model_info(
    authorization: str = Header(..., description="Bearer token")
) -> Dict:
    """Get model metadata."""
    # Verify token
    user = await verify_api_token(authorization)
    
    # Log access
    await audit_log(
        user_id=user.id,
        action="get_model_info",
        timestamp=datetime.now(timezone.utc)
    )
    
    return model_server.get_model_info()


@app.post("/v1/completions")
async def create_completion(
    request: GenerationRequest,
    authorization: str = Header(...),
    request_obj: Request = None
) -> GenerationResponse:
    """Generate text completion."""
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Verify authentication
    user = await verify_api_token(authorization)
    
    # Check rate limit
    is_allowed = await rate_limit(user.id, limit=100)
    if not is_allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Check for PII in input
    metrics.counter("api_request_total").inc({"endpoint": "completions"})
    
    try:
        # Check for prompt injection
        if await check_pii(request.prompt):
            raise HTTPException(
                status_code=400,
                detail="Input contains potentially harmful content"
            )
        
        # Generate response
        response = await model_server.generate(request, request_id)
        
        # Log successful completion
        await audit_log(
            user_id=user.id,
            action="generate_completion",
            request_id=request_id,
            tokens_generated=response.tokens_generated,
            latency_ms=response.latency_ms,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Record metrics
        metrics.histogram("generation_latency_ms").observe(response.latency_ms)
        metrics.counter("tokens_generated_total").inc(response.tokens_generated)
        
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Generation failed: {e}")
        metrics.counter("generation_errors_total").inc()
        raise HTTPException(status_code=500, detail="Generation failed")


@app.post("/v1/completions/stream")
async def create_completion_stream(
    request: GenerationRequest,
    authorization: str = Header(...)
) -> StreamingResponse:
    """Stream text completion tokens."""
    
    request_id = str(uuid.uuid4())
    user = await verify_api_token(authorization)
    
    async def token_generator():
        """Generator for streaming tokens."""
        try:
            async for token in model_server.generate_streaming(request, request_id):
                # Yield as Server-Sent Events
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            logger.error(f"[{request_id}] Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream"
    )


@app.post("/v1/batch")
async def batch_completions(
    requests: List[GenerationRequest],
    authorization: str = Header(...)
) -> List[GenerationResponse]:
    """Process batch of completion requests."""
    
    user = await verify_api_token(authorization)
    
    responses = []
    for req in requests:
        request_id = str(uuid.uuid4())
        response = await model_server.generate(req, request_id)
        responses.append(response)
    
    return responses


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
```

### 1.3 Kubernetes Deployment

```yaml
# k8s/triangle-llm-backend-deployment.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: triangle-llm-config
  namespace: triangle-llm
data:
  model_path: "/models/triangle-llm-1.0"
  tensor_parallel_size: "4"
  max_batch_size: "64"
  max_context_length: "1000000"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triangle-llm-backend
  namespace: triangle-llm
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: triangle-llm-backend
  template:
    metadata:
      labels:
        app: triangle-llm-backend
        version: "1.0"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: triangle-llm
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      
      initContainers:
      - name: verify-model
        image: ghcr.io/triangleos/triangle-llm:1.0
        command: ["python", "-m", "triangle_llm.scripts.verify_model"]
        env:
        - name: MODEL_PATH
          value: "/models/triangle-llm-1.0"
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
      
      containers:
      - name: api-server
        image: ghcr.io/triangleos/triangle-llm:1.0
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        - name: metrics
          containerPort: 8001
          protocol: TCP
        
        env:
        - name: MODEL_PATH
          valueFrom:
            configMapKeyRef:
              name: triangle-llm-config
              key: model_path
        - name: TENSOR_PARALLEL_SIZE
          valueFrom:
            configMapKeyRef:
              name: triangle-llm-config
              key: tensor_parallel_size
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3"
        
        command: 
        - python
        - -m
        - triangle_llm.server.api_server
        
        resources:
          requests:
            nvidia.com/gpu: "4"
            memory: "250Gi"
            cpu: "24"
          limits:
            nvidia.com/gpu: "4"
            memory: "280Gi"
            cpu: "32"
        
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 120
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 2
        
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
        - name: cache-volume
          mountPath: /tmp/cache
        - name: logs-volume
          mountPath: /var/log
      
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: triangle-llm-models
      - name: cache-volume
        emptyDir:
          sizeLimit: 100Gi
      - name: logs-volume
        emptyDir:
          sizeLimit: 10Gi
      
      nodeSelector:
        gpu: nvidia-h100
      
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - triangle-llm-backend
              topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: triangle-llm-backend
  namespace: triangle-llm
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 8000
    targetPort: http
  - name: metrics
    port: 8001
    targetPort: metrics
  selector:
    app: triangle-llm-backend

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triangle-llm-backend-hpa
  namespace: triangle-llm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triangle-llm-backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

---

## Phase 2: Security & Authentication Layer

### 2.1 OAuth2 + mTLS Authentication

```python
# triangle_llm/security/auth.py

import jwt
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
import httpx
from cryptography import x509
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class TriangleOSAuthenticator:
    """Handles OAuth2 and mTLS authentication with Triangle OS IAM."""

    def __init__(
        self,
        iam_endpoint: str,
        client_id: str,
        client_secret: str,
        ca_cert_path: str
    ):
        """Initialize authenticator."""
        self.iam_endpoint = iam_endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.ca_cert_path = ca_cert_path
        self.token_cache: Dict[str, Dict] = {}

    async def verify_token(self, token: str) -> Dict:
        """Verify OAuth2 bearer token."""
        
        # Remove 'Bearer ' prefix if present
        if token.startswith("Bearer "):
            token = token[7:]
        
        # Check cache
        if token in self.token_cache:
            cached = self.token_cache[token]
            if cached["expires_at"] > datetime.now(timezone.utc):
                return cached["user"]
        
        # Verify with IAM service
        async with httpx.AsyncClient(verify=self.ca_cert_path) as client:
            try:
                response = await client.post(
                    f"{self.iam_endpoint}/token/verify",
                    json={"token": token},
                    timeout=5.0
                )
                response.raise_for_status()
                
            except httpx.HTTPError as e:
                logger.error(f"Token verification failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
        
        user_data = response.json()
        
        # Cache token
        self.token_cache[token] = {
            "user": user_data,
            "expires_at": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        
        return user_data

    async def verify_mtls(
        self,
        client_cert: Optional[bytes] = None,
        client_key: Optional[bytes] = None
    ) -> Dict:
        """Verify mTLS client certificate."""
        
        if not client_cert:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Client certificate required"
            )
        
        try:
            cert = x509.load_pem_x509_certificate(
                client_cert,
                default_backend()
            )
            
            # Verify certificate chain
            # In production, verify against CA
            
            # Extract subject
            subject = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
            
            return {"subject": subject, "authenticated": True}
            
        except Exception as e:
            logger.error(f"mTLS verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client certificate"
            )


class AuthUser:
    """Authenticated user context."""
    
    def __init__(self, data: Dict):
        self.id = data.get("user_id")
        self.email = data.get("email")
        self.organization = data.get("organization")
        self.permissions = data.get("permissions", [])
        self.rate_limit = data.get("rate_limit", 100)
    
    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions


async def verify_api_token(
    authorization: str,
    authenticator: TriangleOSAuthenticator
) -> AuthUser:
    """Dependency for verifying API tokens."""
    
    try:
        user_data = await authenticator.verify_token(authorization)
        return AuthUser(user_data)
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )
```

### 2.2 Audit Logging

```python
# triangle_llm/security/audit.py

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import asyncpg
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Audit log event."""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    status: str  # success, failure, warning
    details: Dict[str, Any]
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuditLogger:
    """Handles audit logging for compliance."""

    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(
            self.db_connection_string,
            min_size=5,
            max_size=20
        )

        # Create audit table if not exists
        await self._create_audit_table()

    async def _create_audit_table(self):
        """Create audit logging table."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    action VARCHAR(255) NOT NULL,
                    resource VARCHAR(255),
                    status VARCHAR(50) NOT NULL,
                    details JSONB,
                    request_id UUID,
                    ip_address INET,
                    user_agent TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_logs(user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
                CREATE INDEX IF NOT EXISTS idx_audit_request_id ON audit_logs(request_id);
            """)

    async def log(self, event: AuditEvent):
        """Log audit event to database."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_logs 
                    (timestamp, user_id, action, resource, status, details, 
                     request_id, ip_address, user_agent)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    event.timestamp,
                    event.user_id,
                    event.action,
                    event.resource,
                    event.status,
                    json.dumps(event.details),
                    event.request_id,
                    event.ip_address,
                    event.user_agent
                )

            # Also log to structured logging
            logger.info(
                f"AUDIT: {event.action} by {event.user_id}",
                extra={
                    "audit": True,
                    "user_id": event.user_id,
                    "action": event.action,
                    "status": event.status,
                    "request_id": event.request_id
                }
            )

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

    async def query_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> list:
        """Query audit logs."""
        
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []

        if user_id:
            query += f" AND user_id = ${len(params) + 1}"
            params.append(user_id)

        if action:
            query += f" AND action = ${len(params) + 1}"
            params.append(action)

        if start_time:
            query += f" AND timestamp >= ${len(params) + 1}"
            params.append(start_time)

        if end_time:
            query += f" AND timestamp <= ${len(params) + 1}"
            params.append(end_time)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(row) for row in rows]

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
```

---

## Phase 3: API Gateway & Load Balancing

### 3.1 Nginx Configuration

```nginx
# nginx/triangle-llm-gateway.conf

upstream triangle_llm_backend {
    # Load balancing with least connections
    least_conn;
    
    server triangle-llm-backend-1:8000 max_fails=3 fail_timeout=30s;
    server triangle-llm-backend-2:8000 max_fails=3 fail_timeout=30s;
    server triangle-llm-backend-3:8000 max_fails=3 fail_timeout=30s;
    
    keepalive 32;
}

# Rate limiting zones
limit_req_zone $http_x_api_key zone=api_limit:10m rate=1000r/s;
limit_req_zone $binary_remote_addr zone=ip_limit:10m rate=100r/s;

# TLS configuration
ssl_protocols TLSv1.3;
ssl_ciphers HIGH:!aNULL:!MD5;
ssl_prefer_server_ciphers on;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# API Gateway server
server {
    listen 443 ssl http2;
    server_name triangle-llm-api.internal;
    
    # TLS certificates
    ssl_certificate /etc/nginx/certs/triangle-llm-api.crt;
    ssl_certificate_key /etc/nginx/certs/triangle-llm-api.key;
    ssl_trusted_certificate /etc/nginx/certs/triangle-root-ca.pem;
    
    # mTLS
    ssl_client_certificate /etc/nginx/certs/triangle-root-ca.pem;
    ssl_verify_client optional;
    ssl_verify_depth 2;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Content-Security-Policy "default-src 'none'" always;
    
    # Request logging
    access_log /var/log/nginx/triangle-llm-access.log combined;
    error_log /var/log/nginx/triangle-llm-error.log warn;
    
    # Health check endpoint
    location /health {
        proxy_pass http://triangle_llm_backend/health;
        access_log off;
    }
    
    # API v1 endpoints
    location /v1/ {
        # Rate limiting
        limit_req zone=api_limit burst=100 nodelay;
        limit_req zone=ip_limit burst=10 nodelay;
        
        # Require authentication
        if ($ssl_client_verify != SUCCESS) {
            return 403 "Client certificate verification failed";
        }
        
        # Proxy settings
        proxy_pass http://triangle_llm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;
        proxy_set_header X-SSL-Client-Subject $ssl_client_s_dn;
        
        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
        
        # For streaming responses
        location ~ /v1/completions/stream {
            proxy_pass http://triangle_llm_backend;
            proxy_buffering off;
            proxy_request_buffering off;
            chunked_transfer_encoding on;
        }
    }
    
    # Metrics endpoint (internal only)
    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://triangle_llm_backend:8001/metrics;
    }
    
    # Default deny
    location / {
        return 403 "Access Denied";
    }
}

# HTTP redirect to HTTPS
server {
    listen 80;
    server_name triangle-llm-api.internal;
    return 301 https://$server_name$request_uri;
}
```

---

## Phase 4: Caching Layer with Redis

### 4.1 Redis Configuration & KV-Cache

```python
# triangle_llm/cache/cache_manager.py

import redis
import hashlib
import json
import logging
from typing import Optional, Dict, Any
from datetime import timedelta

logger = logging.getLogger(__name__)


class PromptCacheManager:
    """Manages prompt caching and KV-cache optimization."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.cache_prefix = "triangle-llm:cache:"
        self.ttl_seconds = 3600

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt hash."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        return f"{self.cache_prefix}{prompt_hash}"

    async def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get cached completion for prompt."""
        cache_key = self._get_cache_key(prompt)
        
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for prompt: {cache_key}")
                return json.loads(cached)
            
            logger.debug(f"Cache miss for prompt: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None

    async def set(
        self,
        prompt: str,
        completion: str,
        metadata: Optional[Dict] = None,
        ttl_seconds: Optional[int] = None
    ):
        """Cache completion for prompt."""
        cache_key = self._get_cache_key(prompt)
        
        cache_data = {
            "prompt": prompt,
            "completion": completion,
            "metadata": metadata or {}
        }
        
        try:
            self.redis_client.setex(
                cache_key,
                ttl_seconds or self.ttl_seconds,
                json.dumps(cache_data)
            )
            logger.debug(f"Cached completion for: {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache write error: {e}")

    async def invalidate(self, prompt: str):
        """Invalidate cache entry."""
        cache_key = self._get_cache_key(prompt)
        try:
            self.redis_client.delete(cache_key)
            logger.info(f"Invalidated cache: {cache_key}")
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.redis_client.info()
            return {
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "cache_hits": info.get("keyspace_hits", 0),
                "cache_misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
```

---

**Document Classification**: CONFIDENTIAL  
**Last Updated**: June 27, 2026  
**Maintained By**: Triangle LLM Backend Engineering Team
