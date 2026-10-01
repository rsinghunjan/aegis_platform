 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
```markdown
- Each client adds/subtracts pairwise masks to its update so that, when all masked updates are summed,
  masks cancel out and the server recovers the true aggregate without seeing individual updates.

Security & privacy properties (prototype)
- Prevents server from seeing raw individual model updates when every client participates and masks cancel.
- No raw updates are revealed to server if all clients faithfully execute the protocol.
- Compatible with client-side DP (clipping + Gaussian noise) for stronger formal privacy guarantees.

Critical limitations (must-read)
- Not tolerant to client dropouts. If one or more clients fail to submit their masked update,
  the mask cancellation breaks and aggregate is corrupted.
- No protection against malicious clients that deviate from protocol (Byzantine behavior).
- No robust verification of client key ownership or attestation beyond enrollment; in production require mTLS and attestation.
- The scheme uses pairwise masks derived from X25519; it is a building block but not a complete secure-aggregation protocol.

Recommended production path
1. Adopt a vetted secure-aggregation protocol (Bonawitz et al.) or use established frameworks:
   - Open-source implementations / research libraries (look for maintained projects).
   - Consider integrating Google’s Private Aggregation or OpenMined tools as appropriate.
2. Add dropout resilience:
   - Implement masking shares and server-assisted mask reconstruction for dropped clients.
   - Use consistent per-round randomness and robust share-recovery (Shamir shares or similar).
3. Client attestation & authenticity:
   - Use mTLS client certificates or TPM/SGX-based attestation to ensure keys belong to expected devices.
   - Use short-lived certificates minted by a trusted CA (Vault PKI).
4. Auditability & accountability:
   - Log enrollment and revocation events in the secure audit store.
   - Monitor aggregate-level diagnostics and anomalies (large jumps, poisoned updates).
5. Rigorous testing:
   - Simulate dropouts and malicious clients in staging to validate recovery and robustness.
   - Engage a security review of the protocol and threat model.

Operational notes (pilot)
- Use small, controlled cohorts of clients in pilot (e.g., < 50) where dropouts are unlikely.
- If dropouts are expected, run rounds only when a quorum of clients is available.
- Keep DP noise enabled for additional privacy protection and to reduce sensitivity to single-client contributions.

References
- Bonawitz, K. et al., "Practical Secure Aggregation for Federated Learning at Scale"
- Flower framework: https://flower.dev/
- OpenMined: https://www.openmined.org/
- TensorFlow Federated / TF-Privacy / Opacus for DP building blocks
docs/runbook_federated_secure_aggregation.md
