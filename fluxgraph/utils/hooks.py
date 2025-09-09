# fluxgraph/utils/hooks.py
class EventHooks:
    """
    (MVP Stub) Transparent debugging and execution tracking.
    """
    def __init__(self):
        self._hooks = {}

    def on(self, event: str, callback):
        """Register a callback for an event."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def trigger(self, event: str, data):
        """Trigger an event."""
        for callback in self._hooks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                print(f"Hook error for '{event}': {e}")

hooks = EventHooks() # Global instance
