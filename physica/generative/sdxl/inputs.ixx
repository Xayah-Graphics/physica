export module physica.generative.sdxl.inputs;
import std;

export namespace physica::generative::sdxl {
    struct Tokens final {
        std::vector<std::int32_t> ids;
        std::vector<float> weights;
        int chunks{};
    };
    // ASCII prompts; HTML entities are literal. Scratch and word cache belong to the model.
    struct Tokenizer final {
        static constexpr std::string_view implementation    = "physica_clip_ascii_bpe_v1";
        static constexpr std::string_view vocabulary_sha256 = "a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4";
        Tokens encode(std::string_view text, std::int32_t padding);

    private:
        struct Symbol {
            std::uint16_t token;
            int previous, next;
        };
        struct Merge {
            std::uint32_t pair;
            int position;
            std::uint16_t rank, token;
            bool operator>(const Merge& other) const;
        };
        struct WordHash {
            struct is_transparent {};
            std::size_t operator()(std::string_view word) const;
        };

        void tokenize(std::string_view text);
        void bpe(std::string_view word);
        void enqueue(int position);

        std::vector<std::pair<std::string_view, double>> segments;
        std::string normalized;
        std::vector<std::int32_t> group;
        std::vector<Symbol> symbols;
        std::vector<Merge> heap;
        std::unordered_map<std::string, std::vector<std::int32_t>, WordHash, std::equal_to<>> cache;
    };
} // namespace physica::generative::sdxl
