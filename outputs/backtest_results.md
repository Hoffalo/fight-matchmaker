# Backtest results — top-3 per event (LogReg baseline)

- Train cutoff: events before **2026-01-01**
- Train size: **508** fights (122 positive, 24.0%)
- Test size: **103** fights across **8** events (26 positive)
- **Model top-3 hit rate: 88%** (7/8 events had a bonus fight in the model's top 3)
- Random top-3 hit rate (analytical baseline): **62%** — model uplift: **+25.7 pp**

> Caveat: only 8 events in the held-out 2026 test window. Per-event hit rate is noisy at this sample size; treat as illustrative.

| Date | Event | Top 3 picks | Actual bonus fights | Hit |
|------|-------|---------------|--------------------|-----|
| 2026-01-24 | UFC 324: Gaethje vs. Pimblett | Josh Hokit vs Denzel Freeman (p=0.57)<br>Natalia Silva vs Rose Namajunas (p=0.55)<br>Ateba Gautier vs Andrey Pulyaev (p=0.54) | Justin Gaethje vs Paddy Pimblett<br>Josh Hokit vs Denzel Freeman<br>Adam Fugitt vs Ty Miller | ✅ |
| 2026-01-31 | UFC 325: Volkanovski vs. Lopes 2 | Rafael Fiziev vs Mauricio Ruffy (p=0.56)<br>Jonathan Micallef vs Oban Elliott (p=0.54)<br>Alexander Volkanovski vs Diego Lopes (p=0.54) | Alexander Volkanovski vs Diego Lopes<br>Rafael Fiziev vs Mauricio Ruffy<br>Quillan Salkilld vs Jamie Mullarkey | ✅ |
| 2026-02-07 | UFC Fight Night: Bautista vs. Oliveira | Jailton Almeida vs Rizvan Kuniev (p=0.57)<br>Wang Cong vs Eduarda Moura (p=0.57)<br>Muin Gafurov vs Jakub Wiklacz (p=0.57) | Mario Bautista vs Vinicius Oliveira<br>Michal Oleksiejczuk vs Marc-Andre Barriault<br>Muin Gafurov vs Jakub Wiklacz | ✅ |
| 2026-02-21 | UFC Fight Night: Strickland vs. Hernandez | Alden Coria vs Luis Gurule (p=0.55)<br>Dan Ige vs Melquizael Costa (p=0.55)<br>Zach Reese vs Michel Pereira (p=0.54) | Sean Strickland vs Anthony Hernandez<br>Geoff Neal vs Uros Medic<br>Dan Ige vs Melquizael Costa<br>Jacobe Smith vs Josiah Harrell | ✅ |
| 2026-02-28 | UFC Fight Night: Moreno vs. Kavanagh | Santiago Luna vs Angel Pacheco (p=0.57)<br>Brandon Moreno vs Lone'er Kavanagh (p=0.57)<br>Regina Tarin vs Ernesta Kareckaite (p=0.57) | Brandon Moreno vs Lone'er Kavanagh<br>Imanol Rodriguez vs Kevin Borjas<br>Regina Tarin vs Ernesta Kareckaite | ✅ |
| 2026-03-07 | UFC 326: Holloway vs. Oliveira 2 | Sumudaerji vs Jesus Aguilar (p=0.55)<br>Ricky Turcios vs Alberto Montes (p=0.55)<br>Donte Johnson vs Cody Brundage (p=0.54) | Drew Dober vs Michael Johnson<br>Gregory Rodrigues vs Brunno Ferreira<br>Ricky Turcios vs Alberto Montes<br>Luke Fernandez vs Rodolfo Bellato | ✅ |
| 2026-03-14 | UFC Fight Night: Emmett vs. Vallejos | Charles Johnson vs Bruno Silva (p=0.58)<br>Vitor Petrino vs Steven Asplund (p=0.58)<br>Piera Rodriguez vs Sam Hughes (p=0.55) | Josh Emmett vs Kevin Vallejos<br>Marwan Rahiki vs Harry Hardwick<br>Bolaji Oki vs Manoel Sousa | ❌ |
| 2026-03-21 | UFC Fight Night: Evloev vs. Murphy | Nathaniel Wood vs Losene Keita (p=0.58)<br>Shem Rock vs Abdul-Kareem Al-Selwady (p=0.53)<br>Shanelle Dyer vs Ravena Oliveira (p=0.52) | Iwo Baraniewski vs Austen Lane<br>Mason Jones vs Axel Sola<br>Shanelle Dyer vs Ravena Oliveira | ✅ |
