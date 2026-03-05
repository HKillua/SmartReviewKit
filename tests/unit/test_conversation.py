"""Unit tests for ConversationStore."""

import pytest

from src.agent.conversation import MemoryConversationStore
from src.agent.types import Message


@pytest.fixture
def store():
    return MemoryConversationStore()


@pytest.mark.asyncio
async def test_create_conversation(store):
    conv = await store.create("user1")
    assert conv.user_id == "user1"
    assert len(conv.messages) == 0


@pytest.mark.asyncio
async def test_get_existing(store):
    conv = await store.create("user1")
    loaded = await store.get(conv.id, "user1")
    assert loaded is not None
    assert loaded.id == conv.id


@pytest.mark.asyncio
async def test_get_wrong_user(store):
    conv = await store.create("user1")
    loaded = await store.get(conv.id, "user2")
    assert loaded is None


@pytest.mark.asyncio
async def test_get_nonexistent(store):
    loaded = await store.get("nope", "user1")
    assert loaded is None


@pytest.mark.asyncio
async def test_update_and_reload(store):
    conv = await store.create("user1")
    conv.messages.append(Message(role="user", content="hello"))
    await store.update(conv)
    loaded = await store.get(conv.id, "user1")
    assert len(loaded.messages) == 1


@pytest.mark.asyncio
async def test_list_conversations(store):
    await store.create("user1")
    await store.create("user1")
    await store.create("user2")
    convs = await store.list_conversations("user1")
    assert len(convs) == 2
