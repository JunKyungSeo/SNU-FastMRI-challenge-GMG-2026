#!/usr/bin/env bash
# ============================================================
# 목적: 작업이 끝났을 때 Gmail로 알림 메일을 보낸다. (GPU 큐 슬롯 아님)
#
# 실행:  bash notify.sh "제목" "본문(생략 가능)"
#        bash shell12.sh; bash notify.sh "shell12 끝남"        (성공/실패 무관 알림)
#        bash shell12.sh && bash notify.sh "shell12 성공" \
#                        || bash notify.sh "shell12 실패!"     (결과 구분 알림)
#
# 최초 1회 설정: 아래 [설정] 3줄 채우기.
#   앱 비밀번호 발급: https://myaccount.google.com/apppasswords (2단계 인증 필요)
# ============================================================

# ===== [설정] 여기 3줄만 채우면 됨 =====
GMAIL_ADDR="grgrgrgr579@gmail.com"   # 보내는 Gmail 계정 (SMTP 로그인에 쓰임)
APP_PW="vrgvggogkaaivlzt"       # Gmail 앱 비밀번호 (공백 없이 붙여서)
TO_ADDR="bedso@snu.ac.kr, esel343434@snu.ac.kr"      # ★ 받는 이메일 주소는 여기 ★
# ======================================

SUBJECT="${1:-작업 완료}"
BODY="${2:-$(date '+%F %T') 작업이 끝났습니다. (host: $(hostname))}"

NOTIFY_SUBJECT="$SUBJECT" NOTIFY_BODY="$BODY" \
NOTIFY_FROM="$GMAIL_ADDR" NOTIFY_PW="$APP_PW" NOTIFY_TO="$TO_ADDR" \
python3 - <<'EOF'
import os, smtplib
from email.mime.text import MIMEText

msg = MIMEText(os.environ["NOTIFY_BODY"])
msg["Subject"] = "[GPU서버] " + os.environ["NOTIFY_SUBJECT"]
msg["From"] = os.environ["NOTIFY_FROM"]
msg["To"] = os.environ["NOTIFY_TO"]

with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
    s.login(os.environ["NOTIFY_FROM"], os.environ["NOTIFY_PW"])
    s.send_message(msg)
EOF

STATUS=$?
if [ "$STATUS" -eq 0 ]; then
    echo "[NOTIFY] 메일 발송 완료 → $TO_ADDR ($SUBJECT)"
else
    echo "[NOTIFY] 메일 발송 실패 (exit=$STATUS) — 앱 비밀번호/포트(465) 확인"
fi
exit "$STATUS"
