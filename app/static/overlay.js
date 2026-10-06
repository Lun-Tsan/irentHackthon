// 直拍取景框：把官方資料包裡同一個視角的真實範例照片半透明疊在畫面上（ghost），
// 用戶把畫面對成跟範例一樣就好。官方照片就是 iRent 期望的構圖，比手畫的輪廓準。
// 範例圖在 data/samples/（由 scripts/make_samples.py 從 500 組裡產生，不進 git），儀表板沒有官方範例，畫一個框。
const GHOST = { ext_fl: 'fl', ext_fr: 'fr', ext_rl: 'rl', ext_rr: 'rr', int_front: 'int_front', int_rear: 'int_rear' };
const HINT = {
  ext_fl: '對成跟範例一樣：左前大燈、保險桿角、前輪',
  ext_fr: '對成跟範例一樣：右前大燈、保險桿角、前輪',
  ext_rl: '對成跟範例一樣：左後尾燈、保險桿角、後輪',
  ext_rr: '對成跟範例一樣：右後尾燈、保險桿角、後輪',
  int_front: '對成跟範例一樣：方向盤、座椅、腳踏墊',
  int_rear: '對成跟範例一樣：後座椅面和腳踏墊，太暗請開車內燈',
};
window.overlayPortrait = function (s) {
  const ac = 'fill:none;stroke:#e8935a;stroke-width:1.6;stroke-dasharray:2.5 2';
  const T = (t, y = 150, size = 4.2) => `<text x="45" y="${y}" text-anchor="middle" font-size="${size}" fill="#fff" style="paint-order:stroke;stroke:#000;stroke-width:1.4">${t}</text>`;
  if (s.view === 'dash') {
    return `<svg viewBox="0 0 90 160" preserveAspectRatio="xMidYMid slice"><rect style="${ac}" x="16" y="66" width="58" height="26" rx="2"/>${T('儀表板里程數放在橘框內，數字要清楚')}</svg>`;
  }
  const g = GHOST[s.id];
  // data/samples/ is generated locally from the confidential data pack (scripts/make_samples.py); if it is missing, just drop the ghost
  const ghost = g ? `<img class="ghost" src="/files/samples/${g}.jpg" alt="" onerror="this.remove()">` : '';
  return `${ghost}<svg viewBox="0 0 90 160" preserveAspectRatio="xMidYMid slice"><rect x="1" y="1" width="88" height="158" rx="3" style="fill:none;stroke:#fff;stroke-width:.8;stroke-dasharray:3 2;opacity:.7"/>${T(HINT[s.id] || '對成跟範例一樣')}${T('（半透明的是範例照片）', 156, 3.4)}</svg>`;
};
