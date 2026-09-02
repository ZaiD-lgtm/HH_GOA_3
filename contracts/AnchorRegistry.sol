// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title AnchorRegistry
/// @notice Tamper-evident registry of evidence-bundle hashes produced by the
///         hhg3 pipeline. Stores only the hash, never the underlying image or
///         personal data; `uri` carries a short pointer/metadata blob.
contract AnchorRegistry {
    struct Record {
        address submitter;
        uint64 timestamp;
        string uri;
    }

    /// @dev recordHash => first anchoring. First write wins; re-anchoring the
    ///      same hash reverts, so the earliest timestamp is the provable one.
    mapping(bytes32 => Record) public records;

    event Anchored(bytes32 indexed recordHash, address indexed submitter, uint64 timestamp, string uri);

    error AlreadyAnchored(bytes32 recordHash);
    error EmptyHash();

    function anchor(bytes32 recordHash, string calldata uri) external {
        if (recordHash == bytes32(0)) revert EmptyHash();
        if (records[recordHash].submitter != address(0)) revert AlreadyAnchored(recordHash);

        records[recordHash] = Record({
            submitter: msg.sender,
            timestamp: uint64(block.timestamp),
            uri: uri
        });

        emit Anchored(recordHash, msg.sender, uint64(block.timestamp), uri);
    }

    function isAnchored(bytes32 recordHash) external view returns (bool) {
        return records[recordHash].submitter != address(0);
    }
}
