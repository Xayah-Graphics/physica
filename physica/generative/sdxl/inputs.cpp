module;
#include "tokenizer-data.h"

module physica.generative.sdxl.inputs;
import std;

namespace physica::generative::sdxl {
    void weighted_segments(const std::string_view text, const double weight, std::vector<std::pair<std::string_view, double>>& segments) {
        int depth{};
        std::size_t begin{};
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '\\' && i + 1 < text.size() && (text[i + 1] == '(' || text[i + 1] == ')')) {
                ++i;
                continue;
            }
            if (text[i] == '(') {
                if (depth == 0) {
                    if (i > begin) segments.emplace_back(text.substr(begin, i - begin), weight);
                    begin = i + 1;
                }
                ++depth;
            } else if (text[i] == ')') {
                --depth;
                if (depth != 0) continue;
                auto inner          = text.substr(begin, i - begin);
                double inner_weight = weight * 1.1;
                const auto colon    = inner.rfind(':');
                if (colon != std::string_view::npos && colon > 0) {
                    auto suffix = inner.substr(colon + 1);
                    std::size_t position{};
                    if (position < suffix.size() && (suffix[position] == '+' || suffix[position] == '-')) ++position;
                    const auto digits_begin = position;
                    while (position < suffix.size() && suffix[position] >= '0' && suffix[position] <= '9') ++position;
                    bool digits = position != digits_begin;
                    if (position < suffix.size() && suffix[position] == '.') {
                        const auto fractional_begin = ++position;
                        while (position < suffix.size() && suffix[position] >= '0' && suffix[position] <= '9') ++position;
                        digits |= position != fractional_begin;
                    }
                    if (position < suffix.size() && (suffix[position] == 'e' || suffix[position] == 'E')) {
                        ++position;
                        if (position < suffix.size() && (suffix[position] == '+' || suffix[position] == '-')) ++position;
                        const auto exponent_begin = position;
                        while (position < suffix.size() && suffix[position] >= '0' && suffix[position] <= '9') ++position;
                        digits &= position != exponent_begin;
                    }
                    if (digits && position == suffix.size()) {
                        if (suffix.front() == '+') suffix.remove_prefix(1);
                        const auto parsed = std::from_chars(suffix.data(), suffix.data() + suffix.size(), inner_weight);
                        if (parsed.ec == std::errc::result_out_of_range) throw std::out_of_range{"Prompt weight"};
                        inner = inner.substr(0, colon);
                    }
                }
                weighted_segments(inner, inner_weight, segments);
                begin = i + 1;
            }
        }
        if (begin < text.size()) segments.emplace_back(text.substr(begin), weight);
    }

    Tokens Tokenizer::encode(const std::string_view text, const std::int32_t padding) {
        segments.clear();
        weighted_segments(text, 1.0, segments);
        Tokens result;
        result.ids.reserve(77);
        result.weights.reserve(77);
        result.ids.push_back(49406);
        result.weights.push_back(1.0F);
        for (const auto& [segment, weight] : segments) {
            normalized.clear();
            for (std::size_t i = 0; i < segment.size(); ++i) {
                char c = segment[i];
                if (c == '\\' && i + 1 < segment.size() && (segment[i + 1] == '(' || segment[i + 1] == ')')) c = segment[++i];
                if (c == 9 || c == 10 || c == 12 || c == 13 || c == ' ') {
                    if (!normalized.empty() && normalized.back() != ' ') normalized += ' ';
                } else normalized += c;
            }
            group.clear();
            // EOS is an unnormalized added token; BOS is matched after lowercase.
            std::size_t begin{};
            while (begin < normalized.size()) {
                const auto eos = normalized.find("<|endoftext|>", begin);
                const auto end = eos == std::string::npos ? normalized.size() : eos;
                for (std::size_t i = begin; i < end; ++i)
                    if (normalized[i] >= 'A' && normalized[i] <= 'Z') normalized[i] += 'a' - 'A';
                while (begin < end) {
                    const auto found = normalized.find("<|startoftext|>", begin);
                    const auto bos   = found < end ? found : end;
                    tokenize(std::string_view{normalized}.substr(begin, bos - begin));
                    if (bos != end) group.push_back(49406);
                    begin = bos == end ? end : bos + 15;
                }
                if (eos != std::string::npos) group.push_back(49407);
                begin = eos == std::string::npos ? normalized.size() : eos + 13;
            }
            std::size_t offset{};
            while (offset < group.size()) {
                const std::size_t space = 76 - result.ids.size() % 77;
                const bool fits         = group.size() - offset <= space;
                const std::size_t count = fits ? group.size() - offset : group.size() >= 8 ? space : 0;
                for (std::size_t i = 0; i < count; ++i) {
                    result.ids.push_back(group[offset++]);
                    result.weights.push_back(static_cast<float>(weight));
                }
                if (!fits) {
                    result.ids.push_back(49407);
                    result.weights.push_back(1.0F);
                    while (result.ids.size() % 77) {
                        result.ids.push_back(padding);
                        result.weights.push_back(1.0F);
                    }
                    result.ids.push_back(49406);
                    result.weights.push_back(1.0F);
                }
            }
        }
        result.ids.push_back(49407);
        result.weights.push_back(1.0F);
        while (result.ids.size() % 77) {
            result.ids.push_back(padding);
            result.weights.push_back(1.0F);
        }
        result.chunks = static_cast<int>(result.ids.size() / 77);
        return result;
    }

    bool Tokenizer::Merge::operator>(const Merge& other) const {
        return rank != other.rank ? rank > other.rank : position > other.position;
    }

    std::size_t Tokenizer::WordHash::operator()(const std::string_view word) const {
        return std::hash<std::string_view>{}(word);
    }

    void Tokenizer::tokenize(const std::string_view text) {
        for (std::size_t begin = 0; begin < text.size();) {
            const char c = text[begin];
            if (c == ' ') {
                ++begin;
                continue;
            }
            std::size_t end = begin + 1;
            if (c >= 'a' && c <= 'z') {
                while (end < text.size() && text[end] >= 'a' && text[end] <= 'z') ++end;
            } else if (c < '0' || c > '9') {
                const auto tail = text.substr(begin);
                if (tail.starts_with("'s") || tail.starts_with("'t") || tail.starts_with("'m") || tail.starts_with("'d")) ++end;
                else if (tail.starts_with("'re") || tail.starts_with("'ve") || tail.starts_with("'ll")) end += 2;
                else
                    while (end < text.size() && text[end] != ' ' && (text[end] < 'a' || text[end] > 'z') && (text[end] < '0' || text[end] > '9')) ++end;
            }
            bpe(text.substr(begin, end - begin));
            begin = end;
        }
    }

    void Tokenizer::bpe(const std::string_view word) {
        if (const auto found = cache.find(word); found != cache.end()) {
            group.append_range(found->second);
            return;
        }
        const auto output_begin = group.size();
        symbols.resize(word.size());
        heap.clear();
        for (int i = 0; i < static_cast<int>(word.size()); ++i) symbols[i] = {tokenizer_data::initial[static_cast<unsigned char>(word[i])], i - 1, i + 1};
        symbols.back().token = tokenizer_data::terminal[static_cast<unsigned char>(word.back())];
        symbols.back().next  = -1;
        for (int i = 0; i + 1 < static_cast<int>(word.size()); ++i) enqueue(i);
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), std::greater<>{});
            const auto merge = heap.back();
            heap.pop_back();
            auto& left = symbols[merge.position];
            if (left.next == -1) continue;
            auto& right = symbols[left.next];
            if ((std::uint32_t{left.token} << 16 | right.token) != merge.pair) continue;
            left.token = merge.token;
            left.next  = right.next;
            right.next = -1;
            if (left.next != -1) symbols[left.next].previous = merge.position;
            if (left.previous != -1) enqueue(left.previous);
            enqueue(merge.position);
        }
        for (int position = 0; position != -1; position = symbols[position].next) group.push_back(symbols[position].token);
        if (cache.size() < 10000 && word.size() < 256) cache.emplace(std::string{word}, std::vector<std::int32_t>{group.begin() + output_begin, group.end()});
    }

    void Tokenizer::enqueue(const int position) {
        const auto& left = symbols[position];
        if (left.next == -1) return;
        const std::uint32_t pair = std::uint32_t{left.token} << 16 | symbols[left.next].token;
        std::uint32_t slot       = (pair * 0x9e3779b1u) >> 15;
        while (tokenizer_data::merges[slot].rank != 65535) {
            const auto& rule = tokenizer_data::merges[slot];
            if (rule.pair == pair) {
                heap.push_back({pair, position, rule.rank, rule.token});
                std::push_heap(heap.begin(), heap.end(), std::greater<>{});
                return;
            }
            slot = (slot + 1) & 131071;
        }
    }
} // namespace physica::generative::sdxl
