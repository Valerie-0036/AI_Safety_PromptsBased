from __future__ import annotations

import json
import logging
import os  # Added for environment variables
import uuid  # Added for generating unique Redis keys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import yaml
import redis  # Added redis import
from cryptography.fernet import Fernet, InvalidToken  # Added cryptography import

from langchain_experimental.data_anonymizer.base import (
    DEFAULT_DEANONYMIZER_MATCHING_STRATEGY,
    AnonymizerBase,
    ReversibleAnonymizerBase,
)
from langchain_experimental.data_anonymizer.deanonymizer_mapping import (
    DeanonymizerMapping,
    MappingDataType,
    create_anonymizer_mapping,
)
from langchain_experimental.data_anonymizer.deanonymizer_matching_strategies import (
    exact_matching_strategy,
)
from langchain_experimental.data_anonymizer.faker_presidio_mapping import (
    get_pseudoanonymizer_mapping,
)

# New imports required for NlpEngine type hinting and runtime checks for different engines
from presidio_analyzer.nlp_engine import (
    NlpEngine,
    TransformersNlpEngine,
    SpacyNlpEngine,
    NerModelConfiguration,
)

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine, EntityRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import ConflictResolutionStrategy, OperatorConfig

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _import_analyzer_engine() -> "AnalyzerEngine":
    try:
        from presidio_analyzer import AnalyzerEngine

    except ImportError as e:
        raise ImportError(
            "Could not import presidio_analyzer, please install with "
            "`pip install presidio-analyzer`. You will also need to download a "
            "spaCy model to use the analyzer, e.g. "
            "`python -m spacy download en_core_web_lg`."
        ) from e
    return AnalyzerEngine


def _import_nlp_engine_provider() -> "NlpEngineProvider":
    try:
        from presidio_analyzer.nlp_engine import NlpEngineProvider

    except ImportError as e:
        raise ImportError(
            "Could not import presidio_analyzer, please install with "
            "`pip install presidio-analyzer`. You will also need to download a "
            "spaCy model to use the analyzer, e.g. "
            "`python -m spacy download en_core_web_lg`."
        ) from e
    return NlpEngineProvider


def _import_anonymizer_engine() -> "AnonymizerEngine":
    try:
        from presidio_anonymizer import AnonymizerEngine
    except ImportError as e:
        raise ImportError(
            "Could not import presidio_anonymizer, please install with "
            "`pip install presidio-anonymizer`."
        ) from e
    return AnonymizerEngine


def _import_operator_config() -> "OperatorConfig":
    try:
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError as e:
        raise ImportError(
            "Could not import presidio_anonymizer, please install with "
            "`pip install presidio-anonymizer`."
        ) from e
    return OperatorConfig


# Configuring Anonymizer for multiple languages
# Detailed description and examples can be found here:
# langchain/docs/extras/guides/privacy/multi_language_anonymization.ipynb
DEFAULT_LANGUAGES_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_lg"},
    ],
}


class PresidioAnonymizerBase(AnonymizerBase):
    """Base Anonymizer using Microsoft Presidio.

    See more: https://microsoft.github.io/presidio/
    """

    def __init__(
        self,
        analyzed_fields: Optional[List[str]] = None,
        operators: Optional[Dict[str, OperatorConfig]] = None,
        languages_config: Optional[Dict] = None,
        nlp_engine: Optional[NlpEngine] = None,
        add_default_faker_operators: bool = True,
        faker_seed: Optional[int] = None,
    ):
        """
        Args:
            analyzed_fields: List of fields to detect and then anonymize.
                Defaults to all entities supported by Microsoft Presidio.
            operators: Operators to use for anonymization.
                Operators allow for custom anonymization of detected PII.
                Learn more:
                https://microsoft.github.io/presidio/tutorial/10_simple_anonymization/
            languages_config: Configuration for the NLP engine (for Spacy/Stanza via NlpEngineProvider).
                First language in the list will be used as the main language
                in self.anonymize(...) when no language is specified.
                This parameter is ignored if `nlp_engine` is provided.
                Learn more:
                https://microsoft.github.io/presidio/analyzer/customizing_nlp_models/
            nlp_engine: An instance of a Presidio NlpEngine (e.g., TransformersNlpEngine, SpacyNlpEngine).
                If provided, this engine will be used directly, and `languages_config` will be ignored.
            faker_seed: Seed used to initialize faker.
                Defaults to None, in which case faker will be seeded randomly
                and provide random values.
        """
        OperatorConfig = _import_operator_config()
        AnalyzerEngine = _import_analyzer_engine()
        NlpEngineProvider = _import_nlp_engine_provider()
        AnonymizerEngine = _import_anonymizer_engine()

        self.analyzed_fields = (
            analyzed_fields
            if analyzed_fields is not None
            else list(get_pseudoanonymizer_mapping().keys())
        )

        if add_default_faker_operators:
            self.operators = {
                field: OperatorConfig(
                    operator_name="custom", params={"lambda": faker_function}
                )
                for field, faker_function in get_pseudoanonymizer_mapping(
                    faker_seed
                ).items()
            }
        else:
            self.operators = {}

        if operators:
            self.add_operators(operators)

        _nlp_engine_instance: NlpEngine

        if nlp_engine:
            _nlp_engine_instance = nlp_engine
            if isinstance(_nlp_engine_instance, TransformersNlpEngine):
                self.supported_languages = list(set(m["lang_code"] for m in _nlp_engine_instance.models))
            elif isinstance(_nlp_engine_instance, SpacyNlpEngine):
                self.supported_languages = list(_nlp_engine_instance.nlp.keys())
            else:
                raise ValueError(
                    f"Unsupported NlpEngine type: {type(_nlp_engine_instance)}. "
                    "Cannot automatically determine supported languages. "
                    "Please ensure the provided nlp_engine is a known type "
                    "or extend this logic to handle it."
                )
        else:
            if languages_config is None:
                languages_config = DEFAULT_LANGUAGES_CONFIG
            provider = NlpEngineProvider(nlp_configuration=languages_config)
            _nlp_engine_instance = provider.create_engine()
            self.supported_languages = list(_nlp_engine_instance.nlp.keys())

        self._analyzer = AnalyzerEngine(
            supported_languages=self.supported_languages, nlp_engine=_nlp_engine_instance
        )
        self._anonymizer = AnonymizerEngine()


class PresidioAnonymizer(PresidioAnonymizerBase):
    """Anonymizer using Microsoft Presidio."""

    def _anonymize(
        self,
        text: str,
        language: Optional[str] = None,
        allow_list: Optional[List[str]] = None,
        conflict_resolution: Optional[ConflictResolutionStrategy] = None,
    ) -> str:
        """Anonymize text.
        Each PII entity is replaced with a fake value.
        Each time fake values will be different, as they are generated randomly.

        PresidioAnonymizer has no built-in memory -
        so it will not remember the effects of anonymizing previous texts.
        >>> anonymizer = PresidioAnonymizer()
        >>> anonymizer.anonymize("My name is John Doe. Hi John Doe!")
        'My name is Noah Rhodes. Hi Noah Rhodes!'
        >>> anonymizer.anonymize("My name is John Doe. Hi John Doe!")
        'My name is Brett Russell. Hi Brett Russell!'

        Args:
            text: text to anonymize
            language: language to use for analysis of PII
                If None, the first (main) language in the list
                of languages specified in the configuration will be used.
        """
        if language is None:
            language = self.supported_languages[0]
        elif language not in self.supported_languages:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Supported languages are: {self.supported_languages}. "
                "Change your language configuration file to add more languages."
            )

        supported_entities = []
        for recognizer in self._analyzer.get_recognizers(language):
            recognizer_dict = recognizer.to_dict()
            supported_entities.extend(
                [recognizer_dict["supported_entity"]]
                if "supported_entity" in recognizer_dict
                else recognizer_dict["supported_entities"]
            )

        entities_to_analyze = list(
            set(supported_entities).intersection(set(self.analyzed_fields))
        )

        analyzer_results = self._analyzer.analyze(
            text,
            entities=entities_to_analyze,
            language=language,
            allow_list=allow_list,
        )

        filtered_analyzer_results = (
            self._anonymizer._remove_conflicts_and_get_text_manipulation_data(
                analyzer_results, conflict_resolution
            )
        )

        anonymizer_results = self._anonymizer.anonymize(
            text,
            analyzer_results=analyzer_results,
            operators=self.operators,
        )

        anonymizer_mapping = create_anonymizer_mapping(
            text,
            filtered_analyzer_results,
            anonymizer_results,
        )
        return exact_matching_strategy(text, anonymizer_mapping)


class PresidioReversibleAnonymizer(PresidioAnonymizerBase, ReversibleAnonymizerBase):
    """Reversible Anonymizer using Microsoft Presidio."""

    def __init__(
        self,
        analyzed_fields: Optional[List[str]] = None,
        operators: Optional[Dict[str, OperatorConfig]] = None,
        languages_config: Optional[Dict] = None,
        nlp_engine: Optional[NlpEngine] = None,
        add_default_faker_operators: bool = True,
        faker_seed: Optional[int] = None,
        # NEW: Redis and encryption parameters
        redis_host: Optional[str] = None,
        redis_password: Optional[str] = None,
        redis_port: int = 6379,
        encryption_key: Optional[str] = None,
    ):
        super().__init__(
            analyzed_fields,
            operators,
            languages_config,
            nlp_engine,
            add_default_faker_operators,
            faker_seed,
        )
        self._deanonymizer_mapping = DeanonymizerMapping()

        # NEW: Redis client and Fernet for encryption
        self.redis_client: Optional[redis.Redis] = None
        self.fernet: Optional[Fernet] = None

        if redis_host:
            try:
                # If encryption_key is provided, decode_responses should be False to handle raw bytes
                # Otherwise, it can be True for easier handling of plaintext JSON strings.
                decode_responses = False if encryption_key else True
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=0,
                    password=redis_password,
                    encoding='utf-8',
                    encoding_errors='strict',
                    decode_responses=decode_responses
                )
                self.redis_client.ping()
                logger.info(f"Successfully connected to Redis at {redis_host}:{redis_port}.")

                if encryption_key:
                    # Initialize Fernet if encryption key is provided
                    self.fernet = Fernet(encryption_key.encode('utf-8'))
                    logger.info("Fernet encryption initialized.")
                else:
                    logger.warning("Redis host provided but no encryption key. Anonymizer mapping will NOT be encrypted.")

            except redis.RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None # Ensure it's None if connection failed
            except ValueError as e: # Catch error if encryption_key is not valid base64
                logger.error(f"Invalid encryption key provided: {e}")
                self.fernet = None
            except Exception as e:
                logger.error(f"An unexpected error occurred during Redis/Fernet initialization: {e}")
                self.redis_client = None
                self.fernet = None
        else:
            logger.info("Redis host not provided. Redis functionality will be unavailable.")
            if encryption_key:
                 logger.warning("Encryption key provided but no Redis host. Encryption functionality will be unavailable.")

    @property
    def deanonymizer_mapping(self) -> MappingDataType:
        """Return the deanonymizer mapping"""
        return self._deanonymizer_mapping.data

    @property
    def anonymizer_mapping(self) -> MappingDataType:
        """Return the anonymizer mapping
        This is just the reverse version of the deanonymizer mapping."""
        return {
            key: {v: k for k, v in inner_dict.items()}
            for key, inner_dict in self.deanonymizer_mapping.items()
        }

    def _anonymize(
        self,
        text: str,
        language: Optional[str] = None,
        allow_list: Optional[List[str]] = None,
        conflict_resolution: Optional[ConflictResolutionStrategy] = None,
    ) -> str:
        """Anonymize text.
        Each PII entity is replaced with a fake value.
        Each time fake values will be different, as they are generated randomly.
        At the same time, we will create a mapping from each anonymized entity
        back to its original text value.

        Thanks to the built-in memory, all previously anonymised entities
        will be remembered and replaced by the same fake values:
        >>> anonymizer = PresidioReversibleAnonymizer()
        >>> anonymizer.anonymize("My name is John Doe. Hi John Doe!")
        'My name is Noah Rhodes. Hi Noah Rhodes!'
        >>> anonymizer.anonymize("My name is John Doe. Hi John Doe!")
        'My name is Noah Rhodes. Hi Noah Rhodes!'

        Args:
            text: text to anonymize
            language: language to use for analysis of PII
                If None, the first (main) language in the list
                of languages specified in the configuration will be used.
        """
        if language is None:
            language = self.supported_languages[0]

        if language not in self.supported_languages:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Supported languages are: {self.supported_languages}. "
                "Change your language configuration file to add more languages."
            )

        supported_entities = []
        for recognizer in self._analyzer.get_recognizers(language):
            recognizer_dict = recognizer.to_dict()
            supported_entities.extend(
                [recognizer_dict["supported_entity"]]
                if "supported_entity" in recognizer_dict
                else recognizer_dict["supported_entities"]
            )

        entities_to_analyze = list(
            set(supported_entities).intersection(set(self.analyzed_fields))
        )

        analyzer_results = self._analyzer.analyze(
            text,
            entities=entities_to_analyze,
            language=language,
            allow_list=allow_list,
        )

        filtered_analyzer_results = (
            self._anonymizer._remove_conflicts_and_get_text_manipulation_data(
                analyzer_results, conflict_resolution
            )
        )

        anonymizer_results = self._anonymizer.anonymize(
            text,
            analyzer_results=analyzer_results,
            operators=self.operators,
        )

        new_deanonymizer_mapping = create_anonymizer_mapping(
            text,
            filtered_analyzer_results,
            anonymizer_results,
            is_reversed=True,
        )
        self._deanonymizer_mapping.update(new_deanonymizer_mapping)

        return exact_matching_strategy(text, self.anonymizer_mapping)

    def _deanonymize(
        self,
        text_to_deanonymize: str,
        deanonymizer_matching_strategy: Callable[
            [str, MappingDataType], str
        ] = DEFAULT_DEANONYMIZER_MATCHING_STRATEGY,
    ) -> str:
        """Deanonymize text.
        Each anonymized entity is replaced with its original value.
        This method exploits the mapping created during the anonymization process.

        Args:
            text_to_deanonymize: text to deanonymize
            deanonymizer_matching_strategy: function to use to match
                anonymized entities with their original values and replace them.
        """
        if not self._deanonymizer_mapping:
            raise ValueError(
                "Deanonymizer mapping is empty.",
                "Please call anonymize() and anonymize some text first.",
            )

        text_to_deanonymize = deanonymizer_matching_strategy(
            text_to_deanonymize, self.deanonymizer_mapping
        )

        return text_to_deanonymize

    def reset_deanonymizer_mapping(self) -> None:
        """Reset the deanonymizer mapping"""
        self._deanonymizer_mapping = DeanonymizerMapping()

    def save_deanonymizer_mapping(self, file_path: Union[Path, str]) -> None:
        """Save the deanonymizer mapping to a JSON or YAML file.

        Args:
            file_path: Path to file to save the mapping to.

        Example:
        .. code-block:: python

            anonymizer.save_deanonymizer_mapping(file_path="path/mapping.json")
        """

        save_path = Path(file_path)

        if save_path.suffix not in [".json", ".yaml"]:
            raise ValueError(f"{save_path} must have an extension of .json or .yaml")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            with open(save_path, "w") as f:
                json.dump(self.deanonymizer_mapping, f, indent=2)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with open(save_path, "w") as f:
                yaml.dump(self.deanonymizer_mapping, f, default_flow_style=False)

    def load_deanonymizer_mapping(self, file_path: Union[Path, str]) -> None:
        """Load the deanonymizer mapping from a JSON or YAML file.

        Args:
            file_path: Path to file to load the mapping from.

        Example:
        .. code-block:: python

            anonymizer.load_deanonymizer_mapping(file_path="path/mapping.json")
        """

        load_path = Path(file_path)

        if load_path.suffix not in [".json", ".yaml"]:
            raise ValueError(f"{load_path} must have an extension of .json or .yaml")

        if load_path.suffix == ".json":
            with open(load_path, "r") as f:
                loaded_mapping = json.load(f)
        elif load_path.suffix.endswith((".yaml", ".yml")):
            with open(load_path, "r") as f:
                loaded_mapping = yaml.load(f, Loader=yaml.FullLoader)

        self._deanonymizer_mapping.update(loaded_mapping)

    def save_deanonymizer_mapping_to_redis(self, redis_key: Optional[str] = None, ttl_seconds: Optional[int] = None) -> str:
        """Save the current deanonymizer mapping to Redis, encrypted if a key is provided.

        Args:
            redis_key: An optional key to use for storing the mapping in Redis.
                       If None, a UUID will be generated.
            ttl_seconds: Optional time-to-live for the Redis key in seconds.

        Returns:
            The Redis key under which the mapping is stored.

        Raises:
            ValueError: If Redis client is not configured.
            RuntimeError: If encryption fails or Redis operation fails.
        """
        if not self.redis_client:
            raise ValueError("Redis client is not initialized. Please provide redis_host during initialization.")

        if not redis_key:
            redis_key = str(uuid.uuid4())
            logger.info(f"Generated new Redis key: {redis_key}")
        
        # Serialize mapping to JSON string
        mapping_json_str = json.dumps(self.deanonymizer_mapping)
        mapping_bytes = mapping_json_str.encode('utf-8')

        try:
            if self.fernet:
                # Encrypt the bytes
                encrypted_data = self.fernet.encrypt(mapping_bytes)
                self.redis_client.set(redis_key, encrypted_data, ex=ttl_seconds)
                logger.info(f"Encrypted anonymizer mapping saved to Redis under key: {redis_key}")
            else:
                # Save plaintext if encryption not enabled.
                # The Redis client for plaintext is configured with decode_responses=True,
                # so it expects/returns strings.
                self.redis_client.set(redis_key, mapping_json_str, ex=ttl_seconds)
                logger.warning(f"Plaintext anonymizer mapping saved to Redis under key: {redis_key} (encryption not configured).")

            return redis_key
        except redis.RedisError as e:
            logger.error(f"Failed to save anonymizer mapping to Redis: {e}")
            raise RuntimeError(f"Redis save failed: {e}")
        except Exception as e:
            logger.error(f"Failed to encrypt and save anonymizer mapping to Redis: {e}")
            raise RuntimeError(f"Encryption/Redis save failed: {e}")

    def load_deanonymizer_mapping_from_redis(self, redis_key: str) -> None:
        """Load the deanonymizer mapping from Redis, decrypting if configured.

        Args:
            redis_key: The Redis key under which the mapping is stored.

        Raises:
            ValueError: If Redis client is not configured or mapping not found.
            RuntimeError: If decryption fails or Redis operation fails.
        """
        if not self.redis_client:
            raise ValueError("Redis client is not initialized. Please provide redis_host during initialization.")

        try:
            retrieved_data = self.redis_client.get(redis_key)
            if retrieved_data is None:
                raise ValueError(f"No anonymizer mapping found for key: {redis_key}")

            mapping_bytes = None
            if self.fernet:
                # Decrypt the bytes
                mapping_bytes = self.fernet.decrypt(retrieved_data)
                logger.info(f"Encrypted anonymizer mapping loaded and decrypted for key: {redis_key}")
            else:
                # Assume plaintext if encryption not enabled.
                # retrieved_data will be a string because decode_responses=True for plaintext client.
                mapping_bytes = retrieved_data.encode('utf-8') # Convert to bytes for json.loads

                logger.warning(f"Plaintext anonymizer mapping loaded for key: {redis_key} (encryption not configured).")
            
            # Deserialize JSON bytes to dict
            loaded_mapping = json.loads(mapping_bytes.decode('utf-8'))
            self._deanonymizer_mapping.update(loaded_mapping)

        except InvalidToken as e:
            logger.error(f"Failed to decrypt data for Redis key {redis_key}: Invalid encryption token. This can happen if the key is wrong or data is corrupted. Error: {e}")
            raise RuntimeError(f"Decryption failed for key {redis_key}: Invalid encryption token.")
        except redis.RedisError as e:
            logger.error(f"Failed to load anonymizer mapping from Redis: {e}")
            raise RuntimeError(f"Redis load failed: {e}")
        except Exception as e:
            logger.error(f"Failed to decrypt and load anonymizer mapping from Redis: {e}")
            raise RuntimeError(f"Decryption/Redis load failed: {e}")