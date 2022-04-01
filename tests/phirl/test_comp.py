def test_protocol() -> None:

     from phirl._comp import Protocol

     class Interface(Protocol):
         def a(self) -> int:
             pass

     class Implementation(Interface):
         def a(self) -> int:
             return 3

     x = Implementation()
     assert x.a() == 3

