# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Common Voice Dataset"""


import csv
import os
import urllib

import datasets
import requests
from datasets.utils.py_utils import size_str
from huggingface_hub import HfApi, HfFolder

from .release_stats import STATS
LANGUAGES = {'ab': 'Abkhaz', 'ace': 'Acehnese', 'ady': 'Adyghe', 'af': 'Afrikaans', 'am': 'Amharic', 'an': 'Aragonese', 'ar': 'Arabic', 'arn': 'Mapudungun', 'as': 'Assamese', 'ast': 'Asturian', 'az': 'Azerbaijani', 'ba': 'Bashkir', 'bas': 'Basaa', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bn': 'Bengali', 'br': 'Breton', 'bs': 'Bosnian', 'bxr': 'Buryat', 'ca': 'Catalan', 'cak': 'Kaqchikel', 'ckb': 'Central Kurdish', 'cnh': 'Hakha Chin', 'co': 'Corsican', 'cs': 'Czech', 'cv': 'Chuvash', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'dsb': 'Sorbian, Lower', 'dv': 'Dhivehi', 'dyu': 'Dioula', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'ff': 'Fulah', 'fi': 'Finnish', 'fo': 'Faroese', 'fr': 'French', 'fy-NL': 'Frisian', 'ga-IE': 'Irish', 'gl': 'Galician', 'gn': 'Guarani', 'gom': 'Goan Konkani', 'ha': 'Hausa', 'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian', 'hsb': 'Sorbian, Upper', 'ht': 'Haitian', 'hu': 'Hungarian', 'hy-AM': 'Armenian', 'hyw': 'Armenian Western', 'ia': 'Interlingua', 'id': 'Indonesian', 'ie': 'Interlingue', 'ig': 'Igbo', 'is': 'Icelandic', 'it': 'Italian', 'izh': 'Izhorian', 'ja': 'Japanese', 'ka': 'Georgian', 'kaa': 'Karakalpak', 'kab': 'Kabyle', 'kbd': 'Kabardian', 'ki': 'Kikuyu', 'kk': 'Kazakh', 'km': 'Khmer', 'kmr': 'Kurmanji Kurdish', 'knn': 'Konkani (Devanagari)', 'ko': 'Korean', 'kpv': 'Komi-Zyrian', 'kw': 'Cornish', 'ky': 'Kyrgyz', 'lb': 'Luxembourgish', 'lg': 'Luganda', 'lij': 'Ligurian', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mai': 'Maithili', 'mdf': 'Moksha', 'mg': 'Malagasy', 'mhr': 'Meadow Mari', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mni': 'Meetei Lon', 'mos': 'Mossi', 'mr': 'Marathi', 'mrj': 'Hill Mari', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'myv': 'Erzya', 'nan-tw': 'Taiwanese (Minnan)', 'nb-NO': 'Norwegian Bokmål', 'nd': 'IsiNdebele (North)', 'ne-NP': 'Nepali', 'nia': 'Nias', 'nl': 'Dutch', 'nn-NO': 'Norwegian Nynorsk', 'nr': 'IsiNdebele (South)', 'nso': 'Northern Sotho', 'nyn': 'Runyankole', 'oc': 'Occitan', 'om': 'Afaan Ormoo', 'or': 'Odia', 'pa-IN': 'Punjabi', 'pap-AW': 'Papiamento (Aruba)', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'quc': "K'iche'", 'quy': 'Quechua Chanka', 'rm-sursilv': 'Romansh Sursilvan', 'rm-vallader': 'Romansh Vallader', 'ro': 'Romanian', 'ru': 'Russian', 'rw': 'Kinyarwanda', 'sah': 'Sakha', 'sat': 'Santali (Ol Chiki)', 'sc': 'Sardinian', 'scn': 'Sicilian', 'shi': 'Shilha', 'si': 'Sinhala', 'sk': 'Slovak', 'skr': 'Saraiki', 'sl': 'Slovenian', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'ss': 'Siswati', 'st': 'Southern Sotho', 'sv-SE': 'Swedish', 'sw': 'Swahili', 'syr': 'Syriac', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'tig': 'Tigre', 'tk': 'Turkmen', 'tl': 'Tagalog', 'tn': 'Setswana', 'tok': 'Toki Pona', 'tr': 'Turkish', 'ts': 'Xitsonga', 'tt': 'Tatar', 'tw': 'Twi', 'ty': 'Tahitian', 'uby': 'Ubykh', 'udm': 'Udmurt', 'ug': 'Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Tshivenda', 'vec': 'Venetian', 'vi': 'Vietnamese', 'vot': 'Votic', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'yue': 'Cantonese', 'zh-CN': 'Chinese (China)', 'zh-HK': 'Chinese (Hong Kong)', 'zh-TW': 'Chinese (Taiwan)', 'zu': 'Zulu'}

_CITATION = """\
@inproceedings{commonvoice:2020,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
  pages = {4211--4215},
  year = 2020
}
"""

_HOMEPAGE = "https://commonvoice.mozilla.org/en/datasets"

_LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"

_API_URL = "https://commonvoice.mozilla.org/api/v1"


class CommonVoiceConfig(datasets.BuilderConfig):
    """BuilderConfig for CommonVoice."""

    def __init__(self, name, version, **kwargs):
        self.language = kwargs.pop("language", None)
        self.release_date = kwargs.pop("release_date", None)
        self.num_clips = kwargs.pop("num_clips", None)
        self.num_speakers = kwargs.pop("num_speakers", None)
        self.validated_hr = kwargs.pop("validated_hr", None)
        self.total_hr = kwargs.pop("total_hr", None)
        self.size_bytes = kwargs.pop("size_bytes", None)
        self.size_human = size_str(self.size_bytes)
        description = (
            f"Common Voice speech to text dataset in {self.language} released on {self.release_date}. "
            f"The dataset comprises {self.validated_hr} hours of validated transcribed speech data "
            f"out of {self.total_hr} hours in total from {self.num_speakers} speakers. "
            f"The dataset contains {self.num_clips} audio clips and has a size of {self.size_human}."
        )
        super(CommonVoiceConfig, self).__init__(
            name=name,
            version=datasets.Version(version),
            description=description,
            **kwargs,
        )


class CommonVoice(datasets.GeneratorBasedBuilder):
    DEFAULT_CONFIG_NAME = "en"
    DEFAULT_WRITER_BATCH_SIZE = 1000

    BUILDER_CONFIGS = [
        CommonVoiceConfig(
            name=lang,
            version=STATS["version"],
            language=LANGUAGES[lang],
            release_date=STATS["date"],
            num_clips=lang_stats["clips"],
            num_speakers=lang_stats["users"],
            validated_hr=float(lang_stats["validHrs"]) if lang_stats["validHrs"] else None,
            total_hr=float(lang_stats["totalHrs"]) if lang_stats["totalHrs"] else None,
            size_bytes=int(lang_stats["size"]) if lang_stats["size"] else None,
        )
        for lang, lang_stats in STATS["locales"].items()
    ]

    def _info(self):
        total_languages = len(STATS["locales"])
        total_valid_hours = STATS["totalValidHrs"]
        description = (
            "Common Voice is Mozilla's initiative to help teach machines how real people speak. "
            f"The dataset currently consists of {total_valid_hours} validated hours of speech "
            f" in {total_languages} languages, but more voices and languages are always added."
        )
        features = datasets.Features(
            {
                "client_id": datasets.Value("string"),
                "path": datasets.Value("string"),
                "audio": datasets.features.Audio(sampling_rate=48_000),
                "sentence": datasets.Value("string"),
                "up_votes": datasets.Value("int64"),
                "down_votes": datasets.Value("int64"),
                "age": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "accent": datasets.Value("string"),
                "locale": datasets.Value("string"),
                "segment": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=description,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=self.config.version,
            # task_templates=[
            #     AutomaticSpeechRecognition(audio_file_path_column="path", transcription_column="sentence")
            # ],
        )

    def _get_bundle_url(self, locale, url_template):
        # path = encodeURIComponent(path)
        path = url_template.replace("{locale}", locale)
        path = urllib.parse.quote(path.encode("utf-8"), safe="~()*!.'")
        # use_cdn = self.config.size_bytes < 20 * 1024 * 1024 * 1024
        # response = requests.get(f"{_API_URL}/bucket/dataset/{path}/{use_cdn}", timeout=10.0).json()
        response = requests.get(f"{_API_URL}/bucket/dataset/{path}", timeout=10.0).json()
        return response["url"]

    def _log_download(self, locale, bundle_version, auth_token):
        whoami = HfApi().whoami(auth_token)
        email = whoami["email"] if "email" in whoami else ""
        payload = {"email": email, "locale": locale, "dataset": bundle_version}
        requests.post(f"{_API_URL}/{locale}/downloaders", json=payload).json()

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        hf_auth_token = dl_manager.download_config.token if hasattr(dl_manager.download_config, "token") else dl_manager.download_config.use_auth_token
        hf_auth_token = hf_auth_token or HfFolder().get_token()
        if hf_auth_token is None:
            raise EnvironmentError(
                "Token is required to load this dataset. Pass a token"
                " by logging in with `huggingface-cli login` or by passing `token=<your_token>`."
            )

        bundle_url_template = STATS["bundleURLTemplate"]
        bundle_version = bundle_url_template.split("/")[0]
        dl_manager.download_config.ignore_url_params = True

        self._log_download(self.config.name, bundle_version, hf_auth_token)
        archive_path = dl_manager.download(self._get_bundle_url(self.config.name, bundle_url_template))
        local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else None

        if self.config.version < datasets.Version("5.0.0"):
            path_to_data = ""
        else:
            path_to_data = "/".join([bundle_version, self.config.name])
        path_to_clips = "/".join([path_to_data, "clips"]) if path_to_data else "clips"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(archive_path),
                    "metadata_filepath": "/".join([path_to_data, "train.tsv"]) if path_to_data else "train.tsv",
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(archive_path),
                    "metadata_filepath": "/".join([path_to_data, "test.tsv"]) if path_to_data else "test.tsv",
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(archive_path),
                    "metadata_filepath": "/".join([path_to_data, "dev.tsv"]) if path_to_data else "dev.tsv",
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="other",
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(archive_path),
                    "metadata_filepath": "/".join([path_to_data, "other.tsv"]) if path_to_data else "other.tsv",
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="invalidated",
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(archive_path),
                    "metadata_filepath": "/".join([path_to_data, "invalidated.tsv"])
                    if path_to_data
                    else "invalidated.tsv",
                    "path_to_clips": path_to_clips,
                },
            ),
        ]

    def _generate_examples(
        self,
        local_extracted_archive,
        archive_iterator,
        metadata_filepath,
        path_to_clips,
    ):
        """Yields examples."""
        data_fields = list(self._info().features.keys())
        metadata = {}
        metadata_found = False
        for path, f in archive_iterator:
            if path == metadata_filepath:
                metadata_found = True
                lines = (line.decode("utf-8") for line in f)
                reader = csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row in reader:
                    # set absolute path for mp3 audio file
                    if not row["path"].endswith(".mp3"):
                        row["path"] += ".mp3"
                    row["path"] = os.path.join(path_to_clips, row["path"])
                    # accent -> accents in CV 8.0
                    if "accents" in row:
                        row["accent"] = row["accents"]
                        del row["accents"]
                    # if data is incomplete, fill with empty values
                    for field in data_fields:
                        if field not in row:
                            row[field] = ""
                    metadata[row["path"]] = row
            elif path.startswith(path_to_clips):
                assert metadata_found, "Found audio clips before the metadata TSV file."
                if not metadata:
                    break
                if path in metadata:
                    result = dict(metadata[path])
                    # set the audio feature and the path to the extracted file
                    path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                    result["audio"] = {"path": path, "bytes": f.read()}
                    # set path to None if the audio file doesn't exist locally (i.e. in streaming mode)
                    result["path"] = path if local_extracted_archive else None

                    yield path, result
