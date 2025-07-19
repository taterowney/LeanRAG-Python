from LeanRAG import Module, dataset_from_json
from LeanRAG.module import get_decls_from_plaintext

# Mathlib = Module("Mathlib", lean_dir="/Users/trowney/Documents/GitHub/test_project")


# dataset = dataset_from_json("dataset_minictx_theorems_only.json", lean_dir="/Users/trowney/Documents/GitHub/test_project", include_dependencies=True)
dataset = dataset_from_json("test.json", lean_dir="/Users/trowney/Documents/GitHub/test_project", include_dependencies=True)

# dataset = dataset_from_json("test.json", lean_dir="/Users/trowney/Documents/GitHub/test_project")



for module in dataset.all_files(include_dependencies=False):
    for decl in module.declarations():
        print(decl.name, decl.informal_statement)
        for d in decl.dependencies:
            print(f"  - {d.module.name} : {d.name}, {d.informal_statement}")


        # print(decl.src)
        # print(decl.initial_proof_state)
        # print(decl.informal_statement)
        print("-"*50)
    print("="*50 + "\n\n")


# raw = """
#
# /-- Computes `J(a | b)` (or `-J(a | b)` if `flip` is set to `true`) given assumptions, by reducing
# `a` to odd by repeated division and then using quadratic reciprocity to swap `a`, `b`. -/
# private def fastJacobiSymAux (a b : ℕ) (flip : Bool) (ha0 : a > 0) : ℤ :=
#   if ha4 : a % 4 = 0 then
#     fastJacobiSymAux (a / 4) b flip
#       (Nat.div_pos (Nat.le_of_dvd ha0 (Nat.dvd_of_mod_eq_zero ha4)) (by decide))
#   else if ha2 : a % 2 = 0 then
#     fastJacobiSymAux (a / 2) b (xor (b % 8 = 3 ∨ b % 8 = 5) flip)
#       (Nat.div_pos (Nat.le_of_dvd ha0 (Nat.dvd_of_mod_eq_zero ha2)) (by decide))
#   else if ha1 : a = 1 then
#     if flip then -1 else 1
#   else if hba : b % a = 0 then
#     0
#   else
#     fastJacobiSymAux (b % a) a (xor (a % 4 = 3 ∧ b % 4 = 3) flip) (Nat.pos_of_ne_zero hba)
# termination_by a
# decreasing_by
#   · exact a.div_lt_self ha0 (by decide)
#   · exact a.div_lt_self ha0 (by decide)
#   · exact b.mod_lt ha0
#
# private theorem fastJacobiSymAux.eq_jacobiSym {a b : ℕ} {flip : Bool} {ha0 : a > 0}
#     (hb2 : b % 2 = 1) (hb1 : b > 1) :
#     fastJacobiSymAux a b flip ha0 = if flip then -J(a | b) else J(a | b) := by
#   induction a using Nat.strongRecOn generalizing b flip with | ind a IH =>
#   unfold fastJacobiSymAux
#   split <;> rename_i ha4
#   · rw [IH (a / 4) (a.div_lt_self ha0 (by decide)) hb2 hb1]
#     simp only [Int.ofNat_ediv, Nat.cast_ofNat, div_four_left (a := a) (mod_cast ha4) hb2]
#   split <;> rename_i ha2
#   · rw [IH (a / 2) (a.div_lt_self ha0 (by decide)) hb2 hb1]
#     simp only [Int.ofNat_ediv, Nat.cast_ofNat, ← even_odd (a := a) (mod_cast ha2) hb2]
#     by_cases h : b % 8 = 3 ∨ b % 8 = 5 <;> simp [h]; cases flip <;> simp
#   split <;> rename_i ha1
#   · subst ha1; simp
#   split <;> rename_i hba
#   · suffices J(a | b) = 0 by simp [this]
#     refine eq_zero_iff.mpr ⟨fun h ↦ absurd (h ▸ hb1) (by decide), ?_⟩
#     rwa [Int.gcd_natCast_natCast, Nat.gcd_eq_left (Nat.dvd_of_mod_eq_zero hba)]
#   rw [IH (b % a) (b.mod_lt ha0) (Nat.mod_two_ne_zero.mp ha2) (lt_of_le_of_ne ha0 (Ne.symm ha1))]
#   simp only [Int.natCast_mod, ← mod_left]
#   rw [← quadratic_reciprocity_if (Nat.mod_two_ne_zero.mp ha2) hb2]
#   by_cases h : a % 4 = 3 ∧ b % 4 = 3 <;> simp [h]; cases flip <;> simp
#
# /-- Computes `J(a | b)` by reducing `b` to odd by repeated division and then using
# `fastJacobiSymAux`. -/
# private def fastJacobiSym (a : ℤ) (b : ℕ) : ℤ :=
#   if hb0 : b = 0 then
#     1
#   else if _ : b % 2 = 0 then
#     if a % 2 = 0 then
#       0
#     else
#       have : b / 2 < b := b.div_lt_self (Nat.pos_of_ne_zero hb0) one_lt_two
#       fastJacobiSym a (b / 2)
#   else if b = 1 then
#     1
#   else if hab : a % b = 0 then
#     0
#   else
#     fastJacobiSymAux (a % b).natAbs b false (Int.natAbs_pos.mpr hab)
#
# @[csimp] private theorem fastJacobiSym.eq : jacobiSym = fastJacobiSym := by
#   ext a b
#   induction b using Nat.strongRecOn with | ind b IH =>
#   unfold fastJacobiSym
#   split_ifs with hb0 hb2 ha2 hb1 hab
#   · rw [hb0, zero_right]
#   · refine eq_zero_iff.mpr ⟨hb0, ne_of_gt ?_⟩
#     refine Nat.le_of_dvd (Int.gcd_pos_iff.mpr (mod_cast .inr hb0)) ?_
#     refine Nat.dvd_gcd (Int.ofNat_dvd_left.mp (Int.dvd_of_emod_eq_zero ha2)) ?_
#     exact Int.ofNat_dvd_left.mp (Int.dvd_of_emod_eq_zero (mod_cast hb2))
#   · dsimp only
#     rw [← IH (b / 2) (b.div_lt_self (Nat.pos_of_ne_zero hb0) one_lt_two)]
#     obtain ⟨b, rfl⟩ := Nat.dvd_of_mod_eq_zero hb2
#     rw [mul_right' a (by decide) fun h ↦ hb0 (mul_eq_zero_of_right 2 h),
#       b.mul_div_cancel_left (by decide), mod_left a 2, Nat.cast_ofNat,
#       Int.emod_two_ne_zero.mp ha2, one_left, one_mul]
#   · rw [hb1, one_right]
#   · rw [mod_left, hab, zero_left (lt_of_le_of_ne (Nat.pos_of_ne_zero hb0) (Ne.symm hb1))]
#   · rw [fastJacobiSymAux.eq_jacobiSym, if_neg Bool.false_ne_true, mod_left a b,
#       Int.natAbs_of_nonneg (a.emod_nonneg (mod_cast hb0))]
#     · exact Nat.mod_two_ne_zero.mp hb2
#     · exact lt_of_le_of_ne (Nat.one_le_iff_ne_zero.mpr hb0) (Ne.symm hb1)
#
# /-- Computes `legendreSym p a` using `fastJacobiSym`. -/
# @[inline, nolint unusedArguments]
# private def fastLegendreSym (p : ℕ) [Fact p.Prime] (a : ℤ) : ℤ := J(a | p)
#
# @[csimp] private theorem fastLegendreSym.eq : legendreSym = fastLegendreSym := by
#   ext p _ a; rw [legendreSym.to_jacobiSym, fastLegendreSym]
#
# end FastJacobi
#
# """
#
# print(get_decls_from_plaintext(raw, "test"))

# for decl in dataset.Mathlib.Geometry.Manifold.IntegralCurve.Basic.declarations():
#     if decl.name == "eventually_hasDerivAt":
#         print(decl.name)
#         print(decl.src)

# raw = """
#
#
# lemma test := by rfl
#
# lemma IsIntegralCurveOn.hasDerivAt (hγ : IsIntegralCurveOn γ v s) {t : ℝ} (ht : t ∈ s)
#     (hsrc : γ t ∈ (extChartAt I (γ t₀)).source) :
#     HasDerivAt ((extChartAt I (γ t₀)) ∘ γ)
#       (tangentCoordChange I (γ t) (γ t₀) (γ t) (v (γ t))) t := by
#   -- turn `HasDerivAt` into comp of `HasMFDerivAt`
#   have hsrc := extChartAt_source I (γ t₀) ▸ hsrc
#   rw [hasDerivAt_iff_hasFDerivAt, ← hasMFDerivAt_iff_hasFDerivAt]
#   apply (HasMFDerivAt.comp t
#     (hasMFDerivAt_extChartAt (I := I) hsrc) (hγ _ ht)).congr_mfderiv
#   rw [ContinuousLinearMap.ext_iff]
#   intro a
#   rw [ContinuousLinearMap.comp_apply, ContinuousLinearMap.smulRight_apply, map_smul,
#     ← ContinuousLinearMap.one_apply (R₁ := ℝ) a, ← ContinuousLinearMap.smulRight_apply,
#     mfderiv_chartAt_eq_tangentCoordChange hsrc]
#   rfl
# """
# print(get_decls_from_plaintext(raw, "Mathlib.Geometry.Manifold.IntegralCurve.Basic"))