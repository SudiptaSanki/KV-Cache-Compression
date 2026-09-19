"""
Synthetic Long-Context Instruction Task Generator.

Constructs controlled synthetic benchmark prompts with:
1. Early task-critical system instructions (with recorded token spans).
2. Long harmless distractors (simulated technical logs, documentation, code discussions).
3. A final query targeting the early instruction.

Follows the exact prompt construction pattern from the IIT Bombay build specification.
"""

import json
import random
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer

TECHNICAL_DISTRACTORS = [
    "The Kubernetes ingress controller routes inbound traffic across service endpoints using round-robin balancing. "
    "Pods that fail the readiness probe are evicted from the active load-balancer pool within five seconds. "
    "Resource limits specify CPU throttling thresholds under cgroups v2, maintaining node cluster stability. ",

    "Network latency measurements indicate an average round-trip time of 14.2 milliseconds across region us-east-1. "
    "Packet fragmentation was observed when MTU values exceeded 1500 bytes over IPSec tunnel interfaces. "
    "TCP window scaling enabled higher throughput across saturated gigabit backhaul links. ",

    "Distributed tracing via OpenTelemetry captures span contexts across microservice boundaries. "
    "Trace IDs are propagated via W3C Trace Context HTTP headers. Sampling rates are adjusted dynamically "
    "based on downstream queue depth and p99 response percentiles to avoid storage exhaustion. ",

    "Post-mortem analysis of yesterday's incident showed an unexpected memory leak in the garbage collector pool. "
    "Old-generation heap space reached 98% utilization before triggering full stop-the-world compaction cycles. "
    "Heap dumps were uploaded to object storage for heap profiler inspection. ",

    "Database replication lag between primary and read-replicas remained under 50 milliseconds during peak traffic. "
    "Write-ahead logs (WAL) segments are archived to S3 buckets every fifteen minutes. "
    "Continuous archiving ensures point-in-time recovery capabilities across rolling 30-day retention windows. ",

    "The frontend asset pipeline compiles TypeScript bundles using Vite with tree-shaking optimizations. "
    "Static assets are deployed to multi-region CDN edge points with strict Cache-Control headers set to max-age=31536000. "
    "Content security policies prevent arbitrary script injection via strict nonce evaluation. ",

    "Continuous integration pipelines execute linting, unit test suites, and container vulnerability scanning. "
    "Docker images are signed with Cosign keys before push to the private container registry. "
    "Automated canary deployments route 5% of production traffic to newly published images for twenty minutes. "
]

INSTRUCTION_TEMPLATES = [
    {
        "domain": "Database Choice",
        "instructions": [
            "1. Security Rule: The database migration directory must NEVER be modified manually.",
            "2. Database Engine Rule: The database engine must be PostgreSQL.",
            "3. Protocol Rule: API endpoints must return schema version v2."
        ],
        "target_instruction": "Database Engine Rule: The database engine must be PostgreSQL.",
        "target_keyword": "PostgreSQL",
        "query_prefix": "Database Engine Rule: The database engine must be"
    },
    {
        "domain": "Port Configuration",
        "instructions": [
            "1. Network Port Rule: Inbound internal traffic must bind to port 8080.",
            "2. Security Rule: SSL termination is handled at the gateway layer.",
            "3. Telemetry Rule: Reject unauthenticated telemetry probes."
        ],
        "target_instruction": "Network Port Rule: Inbound internal traffic must bind to port 8080.",
        "target_keyword": "8080",
        "query_prefix": "Network Port Rule: Inbound internal traffic must bind to port"
    },
    {
        "domain": "Encryption Standard",
        "instructions": [
            "1. Expiry Rule: Session tokens expire after thirty minutes.",
            "2. Encryption Standard Rule: Resting persistent volumes must use AES-256-GCM encryption.",
            "3. Lifecycle Rule: Key rotation cycles occur quarterly."
        ],
        "target_instruction": "Encryption Standard Rule: Resting persistent volumes must use AES-256-GCM encryption.",
        "target_keyword": "AES-256-GCM",
        "query_prefix": "Encryption Standard Rule: Resting persistent volumes must use"
    },
    {
        "domain": "Log Level",
        "instructions": [
            "1. Healthcheck Rule: Healthcheck probes return HTTP 200.",
            "2. Logging Standard Rule: In production environments, set the log level strictly to WARNING.",
            "3. Format Rule: Structured JSON logging format is mandatory."
        ],
        "target_instruction": "Logging Standard Rule: In production environments, set the log level strictly to WARNING.",
        "target_keyword": "WARNING",
        "query_prefix": "Logging Standard Rule: In production environments, set the log level strictly to"
    },
    {
        "domain": "Cache Engine",
        "instructions": [
            "1. Queue Rule: Background workers process jobs from Redis queues.",
            "2. Application Cache Rule: In-memory application caching must strictly use Memcached.",
            "3. Concurrency Rule: Maximum worker concurrency is capped at eight threads."
        ],
        "target_instruction": "Application Cache Rule: In-memory application caching must strictly use Memcached.",
        "target_keyword": "Memcached",
        "query_prefix": "Application Cache Rule: In-memory application caching must strictly use"
    },
    {
        "domain": "Authentication Header",
        "instructions": [
            "1. Authentication Rule: Service-to-service requests must pass authorization via the X-Secret-Key header.",
            "2. Rate Limit Rule: Rate limits apply across client IP subnets.",
            "3. Webhook Rule: Webhook retries use exponential backoff."
        ],
        "target_instruction": "Authentication Rule: Service-to-service requests must pass authorization via the X-Secret-Key header.",
        "target_keyword": "X-Secret-Key",
        "query_prefix": "Authentication Rule: Service-to-service requests must pass authorization via the"
    },
    {
        "domain": "Deployment Region",
        "instructions": [
            "1. Routing Rule: Canary traffic threshold is 10%.",
            "2. Deployment Region Rule: Primary infrastructure must be deployed in the eu-central-1 region.",
            "3. Disaster Recovery Rule: Disaster recovery failovers trigger after three consecutive probe failures."
        ],
        "target_instruction": "Deployment Region Rule: Primary infrastructure must be deployed in the eu-central-1 region.",
        "target_keyword": "eu-central-1",
        "query_prefix": "Deployment Region Rule: Primary infrastructure must be deployed in the"
    },
    {
        "domain": "Retry Limit",
        "instructions": [
            "1. Storage Rule: Dead letter queue holds rejected payloads.",
            "2. Consumer Retry Rule: Failed message consumers must retry at most 5 times before eviction.",
            "3. Token Rule: Idempotency tokens must be verified."
        ],
        "target_instruction": "Consumer Retry Rule: Failed message consumers must retry at most 5 times before eviction.",
        "target_keyword": "5",
        "query_prefix": "Consumer Retry Rule: Failed message consumers must retry at most"
    }
]

def build_synthetic_prompt(
    template: Dict[str, Any],
    target_token_length: int,
    tokenizer: AutoTokenizer,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Builds a single synthetic prompt meeting a target token length.
    Maps character spans of the critical instruction into exact token spans.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # System instruction block
    sys_header = "=== SYSTEM SPECIFICATION ===\n"
    sys_body = "\n".join(template["instructions"]) + "\n\n"
    
    # Target critical instruction string to track
    target_inst = template["target_instruction"]
    
    # Final task query block
    final_task_header = "\n=== REQUIRED SPECIFICATION RECALL ===\n"
    final_task_body = f"{template['query_prefix']}"

    # Calculate token budget for distractors
    base_text = sys_header + sys_body + final_task_header + final_task_body
    base_tokens = len(tokenizer.encode(base_text))
    needed_distractor_tokens = max(0, target_token_length - base_tokens)

    # Assemble distractors
    distractor_blocks = []
    current_tokens = 0
    while current_tokens < needed_distractor_tokens:
        block = rng.choice(TECHNICAL_DISTRACTORS)
        distractor_blocks.append(block)
        current_tokens += len(tokenizer.encode(block))

    distractor_text = "=== CONTEXT & LOGS ===\n" + "\n".join(distractor_blocks) + "\n"

    # Assemble complete prompt
    full_prompt = sys_header + sys_body + distractor_text + final_task_header + final_task_body

    # Locate critical instruction character span
    char_start = full_prompt.find(target_inst)
    char_end = char_start + len(target_inst) if char_start != -1 else -1

    # Map character span to token span using tokenizer offsets
    enc = tokenizer(full_prompt, return_offsets_mapping=True)
    token_start = -1
    token_end = -1
    if char_start != -1 and "offset_mapping" in enc:
        offsets = enc["offset_mapping"]
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end > char_start and token_start == -1:
                token_start = idx
            if tok_start < char_end:
                token_end = idx + 1

    # Fallback if offsets not supported
    if token_start == -1 or token_end == -1:
        prefix_text = full_prompt[:char_start]
        inst_text = full_prompt[char_start:char_end]
        token_start = len(tokenizer.encode(prefix_text))
        token_end = token_start + len(tokenizer.encode(inst_text))

    actual_length = len(enc.input_ids)

    return {
        "domain": template["domain"],
        "target_instruction": target_inst,
        "target_keyword": template["target_keyword"],
        "query": template.get("query_prefix", ""),
        "prompt": full_prompt,
        "target_length": target_token_length,
        "actual_length": actual_length,
        "char_span": (char_start, char_end),
        "token_span": (token_start, token_end)
    }

def generate_benchmark_dataset(
    output_path: str,
    tokenizer_name: str = "EleutherAI/pythia-70m",
    target_lengths: List[int] = [512, 1024, 2048],
    variations_per_length: int = 10,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generates a full dataset of synthetic prompts and writes them to jsonl.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = []
    prompt_id = 0

    for length in target_lengths:
        for var_idx in range(variations_per_length):
            template = INSTRUCTION_TEMPLATES[var_idx % len(INSTRUCTION_TEMPLATES)]
            sample_seed = seed + prompt_id
            sample = build_synthetic_prompt(
                template=template,
                target_token_length=length,
                tokenizer=tokenizer,
                seed=sample_seed
            )
            sample["id"] = prompt_id
            dataset.append(sample)
            prompt_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {len(dataset)} synthetic prompts to {output_path}")
    return dataset

if __name__ == "__main__":
    import os
    os.makedirs("benchmarks", exist_ok=True)
    generate_benchmark_dataset(
        output_path="benchmarks/prompts.jsonl",
        target_lengths=[512, 1024, 2048],
        variations_per_length=8,
        seed=42
    )
