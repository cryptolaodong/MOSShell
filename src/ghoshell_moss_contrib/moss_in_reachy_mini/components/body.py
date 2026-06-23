import json
from pathlib import Path

import huggingface_hub

from ghoshell_common.contracts import LoggerItf

from ghoshell_moss import Message, Text
from ghoshell_moss.contracts import Workspace
from ghoshell_moss.core.concepts.command import CommandTaskResult
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove
from reachy_mini.utils import create_head_pose
from reachy_mini_dances_library import DanceMove
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES

from ghoshell_moss_contrib.moss_in_reachy_mini.audio.speech_sync import wait_for_speech_start
from ghoshell_moss_contrib.moss_in_reachy_mini.moves.head_move import HeadMove


EMOJI_MAP = {
    "😲": "amazed1",
    "😟": "anxiety1",
    "👂": "attentive1",
    "🧏": "attentive2",
    "😐": "boredom1",
    "😴": "boredom2",
    "🧘": "calming1",
    "😊": "cheerful1",
    "🤗": "come1",
    "😕": "confused1",
    "🙄": "contempt1",
    "👀": "curious1",
    "💃": "dance1",
    "🕺": "dance2",
    "💃🏻": "dance3",                       # skin tone variant
    "😞": "displeased1",
    "😤": "displeased2",
    "🤢": "disgusted1",
    "😔": "downcast1",
    "💀": "dying1",
    "⚡️": "electric1",
    "🤩": "enthusiastic1",
    "😃": "enthusiastic2",
    "🥵": "exhausted1",
    "😨": "fear1",
    "😠": "frustrated1",
    "🤬": "furious1",
    "👋": "go_away1",
    "🙏": "grateful1",
    "🤝": "helpful1",
    "✨": "helpful2",
    "⏳": "impatient1",                     # hourglass to avoid dup
    "😣": "impatient2",
    "🤷": "incomprehensible2",
    "😑": "indifferent1",
    "❓": "inquiring1",
    "❔": "inquiring2",
    "🧐": "inquiring3",                   # two‑emoji sequence
    "😡": "irritated1",                    # angry face (distinct from 😤)
    "🤯": "irritated2",                    # exploding head for stronger irritation
    "😆": "laughing1",
    "😄": "laughing2",
    "🥺": "lonely1",
    "😶": "lost1",                         # no mouth – lost for words
    "🥰": "loving1",
    "🙅": "no1",
    "🙅‍♀️": "no_excited1",                 # woman gesturing NO
    "😢": "no_sad1",
    "😬": "oops1",
    "🤦": "oops2",
    "🏆": "proud1",
    "😎": "proud2",
    "😏": "proud3",
    "💢": "rage1",                       # anger symbol
    "😮‍💨": "relief1",                     # face exhaling
    "😌": "relief2",
    "🚫": "reprimand1",
    "🗣️": "reprimand2",
    "🫵": "reprimand3",
    "🙁": "resigned1",                     # slightly frowning face
    "😥": "sad1",                          # sad but relieved face
    "😭": "sad2",
    "😱": "scared1",                       # face screaming in fear
    "🧘‍♀️": "serenity1",                   # woman meditating
    "😳": "shy1",                          # flushed face
    "😪": "sleep1",                        # sleepy face
    "🏅": "success1",
    "🥳": "success2",
    "😯": "surprised1",                    # hushed face
    "😮": "surprised2",
    "🤔": "thoughtful1",                   # thinking face (same as curious1 – but unique key)
    "💭": "thoughtful2",                   # thought balloon
    "🥱": "tired1",                        # yawning face (same as boredom1 – unique key)
    "😖": "uncomfortable1",
    "🧠": "understanding1",                # thumbs up (affirmative)
    "👌": "understanding2",                # OK hand
    "🤨": "uncertain1",
    "🎊": "welcoming1",                   # waving hand light skin tone
    "🎉": "welcoming2",                    # hugging face (different from come1)
    "✅": "yes1",
    "🫤": "yes_sad1",                    # sad face + thumbs up
}


EMOTION_NAME_ALIASES = {
    "happy": "😊",
    "smile": "😊",
    "cheerful": "😊",
    "laugh": "😆",
    "laughing": "😆",
    "sad": "😢",
    "sleep": "😴",
    "sleepy": "😴",
    "thinking": "🤔",
    "think": "🤔",
    "curious": "👀",
    "surprised": "😮",
    "yes": "✅",
    "no": "🙅",
}
EMOTION_MOVE_NAME_TO_EMOJI = {move_name: emoji for emoji, move_name in EMOJI_MAP.items()}


def _normalize_emotion_emoji(value: str | None) -> str:
    value = (value or "").strip()
    if value in EMOJI_MAP:
        return value
    lowered = value.lower()
    if lowered in EMOTION_NAME_ALIASES:
        return EMOTION_NAME_ALIASES[lowered]
    return EMOTION_MOVE_NAME_TO_EMOJI.get(value, "")


def _load_emotions(ws: Workspace, logger: LoggerItf) -> dict:
    storage = ws.configs().sub_storage("reachy_mini_emotions")
    root = Path(storage.abspath())

    if not root.is_dir() or not any(p.suffix == ".json" for p in root.iterdir()):
        logger.info(f"No reachy_mini_emotions sub-storage in {root}, downloading ...")
        huggingface_hub.snapshot_download(
            repo_id="pollen-robotics/reachy-mini-emotions-library",
            repo_type="dataset",
            local_dir=str(root),
        )

    emotions = {}
    for path in root.iterdir():
        name = path.name
        if name.endswith(".json"):
            emotion = name.removesuffix(".json")
            params = json.loads(storage.get(name))
            emotions[emotion] = params
    return emotions


class Body:
    def __init__(self, mini: ReachyMini, ws: Workspace, logger: LoggerItf):
        self.mini = mini
        self.logger = logger
        self._emotions = _load_emotions(ws, logger)

    async def dance(self, name: str):
        if not AVAILABLE_MOVES.get(name):
            return CommandTaskResult(
                observe=False,
                messages=[Message.new(name="__emotion__").with_content(
                    Text(text=f"本轮你生成的dance={name}是错误的，下次记得使用列表里正确的dance")
                )]
            )
        await wait_for_speech_start("dance", self.logger)
        await self.mini.async_play_move(DanceMove(name))
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(),
        ))
        return CommandTaskResult(
            observe=False,
        )

    def dance_docstring(self):
        beat_dur = 60.0 / DanceMove.default_bpm
        dance_docstrings = []
        for name, move in AVAILABLE_MOVES.items():
            func, params, meta = move
            beats = meta.get("default_duration_beats", 4)
            dur = round(beat_dur * beats, 1)
            desc = meta.get("description", "")
            dance_docstrings.append(f"{name}({beats}拍,{dur}s) {desc}")
        header = (
            f"以下dance的执行时长是固定的（内部BPM={DanceMove.default_bpm}），不随歌曲BPM变化。"
            f"每个dance后面标注的秒数就是实际执行时长（不含复位0.5s）。"
            f"必须使用以下列表中的name，严禁使用未定义的舞蹈名。"
            f"如果dance要和一句话同步，命令要放在那句话之前或句中，不要放在整句话之后。"
        )
        return f"{header}\n" + "\n".join(dance_docstrings)

    async def emotion(self, emoji: str = "", name: str = ""):
        emoji = _normalize_emotion_emoji(emoji) or _normalize_emotion_emoji(name)
        emotion_name = EMOJI_MAP.get(emoji, None)
        if not emotion_name:
            return CommandTaskResult(
                observe=False,
                messages=[Message.new(name="__emotion__").with_content(
                    Text(text=f"本轮你生成的emoji={emoji}是错误的，下次记得使用列表里正确的emoji")
                )]
            )
        params = self._emotions.get(emotion_name)
        if not params:
            return CommandTaskResult(
                observe=False,
                messages=[Message.new(name="__emotion__").with_content(
                    Text(text=f"本轮你生成的emoji={emoji}是错误的，下次记得使用列表里正确的emoji")
                )]
            )

        await wait_for_speech_start("emotion", self.logger)
        await self.mini.async_play_move(RecordedMove(move=params)),
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(),
        ))
        return CommandTaskResult(
            observe=False
        )

    def emotion_docstring(self):
        # emotion_docstrings = []
        # for name, params in self._emotions.items():
        #     emotion_docstrings.append(f"{name} ({params.get('description', '')})")
        # return f"Name choices in \n{"\n".join(emotion_docstrings)}\n"
        return (
            f"必须使用以下列表给定的emoji：{','.join(EMOJI_MAP.keys())}；万不可传非列表内的emoji。"
            f"如果emotion要和一句话同步，命令要放在那句话之前或句中，不要放在整句话之后。"
        )
