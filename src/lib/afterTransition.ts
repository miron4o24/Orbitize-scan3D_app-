// Работа, която може да почака, докато завърши анимацията на прехода между екрани (например опресняване при връщане
// от модела): иначе тежката работа на основната нишка я накъсва. Връща функция за отказ.
const TRANSITION_MS = 380; // малко над анимацията на стека в app/_layout.tsx

export function afterTransition(work: () => void): () => void {
  const timer = setTimeout(work, TRANSITION_MS);
  return () => clearTimeout(timer);
}
