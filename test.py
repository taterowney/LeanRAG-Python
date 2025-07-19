from LeanRAG import Module, dataset_from_json
from LeanRAG.module import get_decls_from_plaintext

# Mathlib = Module("Mathlib", lean_dir="/Users/trowney/Documents/GitHub/test_project")


# dataset = dataset_from_json("dataset_minictx_theorems_only.json", lean_dir="/Users/trowney/Documents/GitHub/test_project", include_dependencies=True)
# dataset = dataset_from_json("test.json", lean_dir="/Users/trowney/Documents/GitHub/test_project", include_dependencies=True)
dataset = dataset_from_json("dataset_minictx_theorems_only.json", lean_dir="/home/trowney/ImProver", include_dependencies=True)


for module in dataset.all_files(include_dependencies=False):
    print(module.name)
    print(module.get_toplevel().name)

    for decl in module.declarations():
        print(decl.name, decl.src, decl.informal_statement)
    break

# for module in dataset.all_files(include_dependencies=False):
#     for decl in module.declarations():
#         print(decl.name, decl.informal_statement)
#         for d in decl.dependencies:
#             print(f"  - {d.module.name} : {d.name}, {d.informal_statement}")
#         print("-"*50)
#     print("="*50 + "\n\n")



# raw = """
#
#
# @[to_additive]
# theorem mul_left_cancel_iff : a * b = a * c ↔ b = c :=
#   ⟨mul_left_cancel, congrArg _⟩
#
# @[to_additive]
# theorem mul_right_injective (a : G) : Injective (a * ·) := fun _ _ ↦ mul_left_cancel
#
# @[to_additive (attr := simp)]
# theorem mul_right_inj (a : G) {b c : G} : a * b = a * c ↔ b = c :=
#   (mul_right_injective a).eq_iff
#
# @[to_additive]
# theorem mul_ne_mul_right (a : G) {b c : G} : a * b ≠ a * c ↔ b ≠ c :=
#   (mul_right_injective a).ne_iff
#
# end IsLeftCancelMul
#
# section IsRightCancelMul
#
# variable [IsRightCancelMul G] {a b c : G}
#
# @[to_additive]
# theorem mul_right_cancel : a * b = c * b → a = c :=
#   IsRightCancelMul.mul_right_cancel a b c
#
# @[to_additive]
# theorem mul_right_cancel_iff : b * a = c * a ↔ b = c :=
#   ⟨mul_right_cancel, congrArg (· * a)⟩
#
# @[to_additive]
# theorem mul_left_injective (a : G) : Function.Injective (· * a) := fun _ _ ↦ mul_right_cancel
#
# @[to_additive (attr := simp)]
# theorem mul_left_inj (a : G) {b c : G} : b * a = c * a ↔ b = c :=
#   (mul_left_injective a).eq_iff
#
# @[to_additive]
# theorem mul_ne_mul_left (a : G) {b c : G} : b * a ≠ c * a ↔ b ≠ c :=
#   (mul_left_injective a).ne_iff
#
# end IsRightCancelMul
# """
# print(get_decls_from_plaintext(raw, "Mathlib.Geometry.Manifold.IntegralCurve.Basic"))