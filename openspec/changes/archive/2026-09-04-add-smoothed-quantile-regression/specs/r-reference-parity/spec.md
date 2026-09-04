## ADDED Requirements

### Requirement: Independent implementations must not borrow incompatible code

Where a reference implementation is licensed incompatibly with this package,
its source SHALL NOT be copied or adapted. Such a capability SHALL be
implemented from published descriptions of the algorithm, validated against the
reference's *outputs*, and the situation SHALL be recorded in the credits.

#### Scenario: A GPL reference for an MIT package

- **WHEN** a capability is based on a reference implementation under a licence
  that this package cannot accept
- **THEN** the implementation is written from the published algorithm, its
  agreement with the reference is demonstrated against outputs rather than
  source, and the credits state that no code was taken

#### Scenario: Numerical agreement is still required

- **WHEN** such an independent implementation is compared with its reference on
  the same inputs
- **THEN** it agrees to documented tolerance, including at extreme quantile
  levels

**Rationale:** independence is about provenance, not accuracy — an independent
implementation is held to the same numerical bar as a port.
