import pytest

from api.memory import ConversationMemory, PersistentConversationMemory


def test_conversation_memory_compatibility_is_bounded():
    memory = ConversationMemory(capacity=2)
    memory.add({"role": "user", "content": "one"})
    memory.add({"role": "assistant", "content": "two"})
    memory.add({"role": "user", "content": "three"})
    assert "one" not in memory.get_conversation()
    assert "three" in memory.get_conversation()


def test_persistent_memory_is_tenant_scoped_and_compacted(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'memory.db'}"
    memory = PersistentConversationMemory(
        "session-1", "tenant-a", capacity=2, database_url=database_url
    )
    memory.add({"role": "user", "content": "first"})
    memory.add({"role": "assistant", "content": "second"})
    memory.add({"role": "user", "content": "third"})

    messages = memory.get_messages()
    assert len(messages) <= 2
    assert "third" in memory.get_conversation(max_chars=100)
    assert "first" in memory.get_conversation(max_chars=100)
    with pytest.raises(PermissionError):
        memory.get_messages(tenant_id="tenant-b")

    other_tenant = PersistentConversationMemory(
        "session-1", "tenant-b", database_url=database_url
    )
    assert other_tenant.get_messages() == []
