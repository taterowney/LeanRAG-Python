from .module import Module, Declaration

Mathlib = Module("Mathlib")
for decl in Mathlib.Algebra.AddConstMap.Basic.declarations():
    print("Informal Statement:", decl.informal_statement)
    print("Informal Proof:", decl.informal_proof)

