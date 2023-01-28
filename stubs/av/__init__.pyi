def open(file, format: Optional[str] = None) -> InputContainer: ...

class Container:
    def __enter__(self) -> "Container":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
