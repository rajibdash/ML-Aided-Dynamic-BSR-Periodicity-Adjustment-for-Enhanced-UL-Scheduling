# 6G Scheduling: Contention-Based Mechanisms for Buffer-Status Reports and Small Data Transmissions

## 1. Executive Summary & The Latency Paradigm Shift
In legacy cellular architectures (4G LTE and 5G NR), uplink (UL) scheduling fundamentally relies on a deterministic, **grant-based (GB)** framework. While highly efficient for high-throughput, continuous traffic, the standard four-step handshake (**Scheduling Request [SR] → Dynamic Grant → Buffer Status Report [BSR] → Uplink Data Allocation**) introduces a structural latency floor. 

As the industry transitions toward **6G applications**—such as multi-sensory Extended Reality (XR), holographic telepresence, and sub-millisecond industrial control systems—this traditional cycle becomes a primary bottleneck. The signaling overhead consumed by tiny, intermittent data packets or frequent buffer updates creates unacceptable control plane congestion and severe packet delay variation (jitter).

To break this bottleneck, **6G architectures shift toward native Contention-Based (CB) or Grant-Free (GF) mechanisms** for both BSR reporting and Small Data Transmission (SDT). By allowing User Equipments (UEs) to autonomously transmit data or buffer states on shared radio resources without waiting for dedicated base station (gNB/6G NodeB) scheduling, the handshake latency is completely eliminated.

---

## 2. Structural Comparison: Grant-Based vs. Contention-Based Uplink

| Operational Metric | Traditional Grant-Based (GB) Loop | 6G Contention-Based (CB) / Grant-Free (GF) |
| :--- | :--- | :--- |
| **Handshake Procedure** | 4-Step: SR $\rightarrow$ Grant $\rightarrow$ BSR $\rightarrow$ UL Assignment | 1-Step or 2-Step Direct Transmission |
| **Control Plane Overhead** | **High**: Dedicated PUCCH for SR, PDCCH for DCI grants. | **Low**: Signaling embedded directly into shared physical channels. |
| **Structural Latency Floor** | Natively bound by frame/slot alignment (typically $\ge 3{-}5\text{ ms}$). | **Sub-millisecond**: Zero-waiting time if resources are available. |
| **Resource Efficiency** | Ideal for high-density, persistent payload streams. | Optimized for sporadic, bursty, time-critical micro-packets. |
| **Collision Profile** | Deterministic; zero collision risk on scheduled slots. | Stochastic; requires advanced collision resolution algorithms. |

---

## 3. Deep-Dive: Contention-Based Mechanisms for BSR Delivery

When packet arrivals are unpredictable, a UE must notify the network of its buffer size immediately. 6G optimizes this using two core contention-based approaches for BSR:

### A. Non-Orthogonal Multiple Access (NOMA) Based BSR
Instead of allocating orthogonal time-frequency slots for individual UEs to send their BSRs, 6G physical layers leverage NOMA.
*   **Signature Assignment:** UEs are assigned unique, non-orthogonal signatures (e.g., specific sparse codes, interleaving patterns, or power-domain layers) from a shared pool.
*   **Simultaneous Transmission:** Multiple UEs experiencing sudden data arrivals transmit their BSRs on the exact same physical resource blocks (PRBs) simultaneously.
*   **Multi-User Detection (MUD):** The gNB utilizes advanced receivers—such as Successive Interference Cancellation (SIC) or Message Passing Algorithms (MPA)—to decode the overlapping buffer reports without requiring prior individual assignments.

### B. RACH-Less Fast-BSR Preamble Mapping
Traditional Random Access Channel (RACH) processes are used for initial access or scheduling recovery. 6G repurposes the preamble space for fast status updates:
*   **Preamble Partitioning:** The available physical RACH (PRACH) preambles are split into dynamic pools corresponding to quantized buffer sizes (e.g., Pool A = low backlog, Pool B = medium backlog, Pool C = critical backlog).
*   **Implicit Signaling:** The UE chooses a preamble from the specific group that matches its current queue state. 
*   **Instant Resource Prediction:** Upon decoding the preamble, the gNB instantly determines not just that a UE wants to transmit, but *exactly how much* data it has waiting, cutting out the explicit MAC-CE BSR transmission step.

---

## 4. Contention-Based Small Data Transmission (SDT) in 6G

For micro-payloads (e.g., TCP ACKs, sensor telemetry, keep-alive pings), transmitting an explicit BSR is inefficient. 6G expands the concepts introduced in 5G NR Rel-17/18 SDT into fully autonomous, high-efficiency contention modes.

```
[UE Side: Packet Arrives in Buffer]
                │
                ▼
   Is Packet Size <= SDT Threshold?
        ├──> YES: Transmit via 2-Step RACH / CG-SDT (Payload + BSR embedded)
        └──> NO:  Trigger fast Contention-Based BSR 
```

### A. Configured Grant SDT (CG-SDT) Optimization
Configured Grants (CG) pre-allocate time-frequency resources. 6G makes these resources **collaboratively shared** among groups of UEs rather than dedicated per user:
*   **Dynamic Resource Pooling:** UEs share a massive, semi-static grid of uplink transmission opportunities.
*   **Piggybacked BSR:** If a packet fits within the pre-allocated transport block (TB), the data is transmitted immediately. If the packet is larger, the data is sent alongside an implicit or explicit BSR in the MAC header, telling the gNB to instantly upgrade the link to a dedicated dynamic grant.

### B. 2-Step Contention-Based RACH (MsgA SDT)
While 5G standardized basic 2-step RACH, 6G scales this by allowing massive transport blocks to be coupled with the initial random-access message:
*   **MsgA Structure:** Combines the physical preamble (for synchronization/channel estimation) and a physical uplink shared channel (PUSCH) payload in a single burst.
*   **Zero-State Overhead:** Even if the UE is in an `INACTIVE` or `IDLE` power-saving state, it can push the small data payload out instantly, avoiding the RRC connection resume signaling cascade.

---

## 5. Collision Mitigation, Resolution, and Advanced Coding

The critical vulnerability of any contention-based system is **packet collision**—when two or more UEs transmit on the same resource slot using identical signatures, causing corruption. 6G mitigates this via distinct architectural layers:

### A. Coded Random Access (CRA)
6G departs from standard slotted ALOHA by implementing **Coded Random Access** (derived from IRSA - Involuntary Repetition Slotted ALOHA). 
*   **Replication:** A UE transmits multiple copies (or coded fragments) of its small data/BSR packet across a block of shared contention slots.
*   **Graph-Based Decoding:** Each packet copy contains pointers to the other slots where its duplicates were sent. When the gNB successfully decodes one clean packet, it uses **Successive Interference Cancellation (SIC)** to subtract that packet's interference waveform from all other slots, iteratively unmasking other collided transmissions like solving a puzzle.

### B. Machine Learning Assisted Contention Windows
Instead of static back-off timers (which add massive jitter when a collision occurs), 6G leverages base station-driven AI to optimize resource availability:
*   **Congestion Prediction:** The gNB uses transformer-based models to monitor historical uplink traffic patterns and predict bursty arrivals (e.g., synchronized frames from an industrial sensor array).
*   **Dynamic Slicing:** The base station dynamically expands or contracts the size of the contention-based physical resource pool in real-time, matching the predicted load to keep the collision probability strictly below a target threshold (e.g., $< 1\%$).

---

## 6. Engineering Trade-offs & Implementation Realities

While contention-based mechanisms offer the holy grail of sub-millisecond latency for small packets, network engineers must manage explicit architectural trade-offs:

1.  **Spectral Efficiency vs. Latency:** Dedicating large pools of physical resource blocks (PRBs) to contention guarantees low latency, but if the network load is low, those resources sit empty, wasting valuable spectrum.
2.  **Base Station Compute Load:** Iterative SIC and multi-user detection algorithms require immense baseband processing power. The gNB must balance the compute cost of decoding collided signals against the radio control overhead saved by skipping the standard grant process.
3.  **Power Consumption:** Frequent contention failures force the UE to retransmit packets at higher power steps, accelerating battery drain for low-power Internet of Things (IoT) devices.

---
*Document Ends.*
