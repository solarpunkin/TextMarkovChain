#include <stdio.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <array>
#include <filesystem>
#include <regex>
#include <iostream>
#include <sstream>

// array hasher
namespace std
{
    template<typename T, size_t N>
    struct hash<array<T, N> >
    {
        typedef array<T, N> argument_type;
        typedef size_t result_type;

        result_type operator()(const argument_type& a) const
        {
            hash<T> hasher;
            result_type h = 0;
            for (result_type i = 0; i < N; ++i)
            {
                h = h * 31 + hasher(a[i]);
            }
            return h;
        }
    };
}

template <typename TType, size_t ORDER_N>
class MarkovChain
{
public:

    MarkovChain()
        : m_rd("dev/random")
        , m_fullSeed{ m_rd(), m_rd(), m_rd(), m_rd(), m_rd(), m_rd(), m_rd(), m_rd() }
        , m_rng(m_fullSeed)
    {

    }

    typedef std::array<TType, ORDER_N> Observations;

    typedef std::unordered_map<TType, size_t> TObservedCounts;

    struct ObservedProbability
    {
        TType observed;
        float cumulativeProbability;
    };

    typedef std::vector<ObservedProbability> TObservedProbabilities;

    struct ObservationContext
    {
        ObservationContext()
        {
            std::fill(m_hasObservation.begin(), m_hasObservation.end(), false);
        }

        Observations              m_observations;
        std::array<bool, ORDER_N> m_hasObservation;
    };

    ObservationContext GetObservationContext()
    {
        return ObservationContext();
    }

    // learning stage
    void RecordObservation(ObservationContext& context, const TType& next)
    {
        // if this observation has a full set of data observed, record this observation
        if (context.m_hasObservation[ORDER_N - 1])
            m_counts[context.m_observations][next]++;

        // move all observations down
        for (size_t index = ORDER_N - 1; index > 0; --index)
        {
            context.m_observations[index] = context.m_observations[index - 1];
            context.m_hasObservation[index] = context.m_hasObservation[index - 1];
        }

        // put in the new observation
        context.m_observations[0] = next;
        context.m_hasObservation[0] = true;
    }

    void FinalizeLearning()
    {
        // turn the sums into cumulative probabilities
        for (auto observed : m_counts)
        {
            size_t sum = 0;
            for (auto nextState : observed.second)
                sum += nextState.second;

            float probabilitySum = 0.0f;
            for (auto nextState : observed.second)
            {
                float probability = float(nextState.second) / float(sum);
                probabilitySum += probability;
                m_probabilities[observed.first].push_back(ObservedProbability{ nextState.first, probabilitySum });
            }
        }
    }

    // data generation stage
    Observations GetInitialObservations()
    {
        // select a starting state entirely at random
        std::uniform_int_distribution<size_t> dist(0, m_probabilities.size());
        size_t index = dist(m_rng);
        auto it = m_probabilities.begin();
        std::advance(it, index);
        return it->first;
    }

    void GetNextObservations(Observations& observations)
    {
        // get the next state by choosing a weighted random next state.
        std::uniform_real_distribution<float> distFloat(0.0f, 1.0f);
        TObservedProbabilities& probabilities = m_probabilities[observations];
        if (probabilities.size() == 0)
            return;

        float nextStateProbability = distFloat(m_rng);
        int nextStateIndex = 0;
        while (nextStateIndex < probabilities.size() - 1 && probabilities[nextStateIndex].cumulativeProbability < nextStateProbability)
            ++nextStateIndex;

        // move all observations down
        for (size_t index = ORDER_N - 1; index > 0; --index)
            observations[index] = observations[index - 1];

        // put the new observation in
        observations[0] = probabilities[nextStateIndex].observed;
    }

    // random number generation storage
    std::random_device m_rd;
    std::seed_seq m_fullSeed;
    std::mt19937 m_rng;

    // data storage
    std::unordered_map<Observations, TObservedCounts> m_counts;
    std::unordered_map<Observations, TObservedProbabilities> m_probabilities;
};

// file output. If you change what type the markov chain works with, you'll have to implement something that handles that type
// like this, for being able to print out the stats file.
template <typename Observations>
void fprintf(FILE* file, const Observations& observations)
{
    bool first = true;
    for (int i = int(observations.size()) - 1; i >= 0; --i)
    {
        if (first)
            ::fprintf(file, "%s", observations[i].c_str());
        else
            ::fprintf(file, " %s", observations[i].c_str());
        first = false;
    }
}

bool IsAlphaNumeric(char c)
{
    if (c >= 'a' && c <= 'z')
        return true;

    if (c >= 'A' && c <= 'Z')
        return true;

    if (c >= '0' && c <= '9')
        return true;

    if (c == '\'')
        return true;

    return false;
}

bool IsPunctuation(char c)
{
    if (c == '.')
        return true;

    if (c == ',')
        return true;

    if (c == ';')
        return true;

    //if (c == '\"')
    //    return true;

    if (c == ':')
        return true;

    if (c == '-')
        return true;

    return false;
}

bool GetWord(unsigned char* contents, size_t size, size_t& position, std::string& word)
{
    // skip ignored characters to start
    while (position < size && !IsAlphaNumeric(contents[position]) && !IsPunctuation(contents[position]))
        position++;

    // exit if there is no word
    if (position >= size)
    {
        word = "";
        return false;
    }

    // go until bad character, or end of data
    size_t startPosition = position;
    if (IsPunctuation(contents[position]))
    {
        while (position < size && IsPunctuation(contents[position]))
            position++;
    }
    else
    {
        while (position < size && IsAlphaNumeric(contents[position]))
            position++;
    }

    // copy the word into the string
    word = std::string(&contents[startPosition], &contents[position]);

    // make lowercase for consistency
    std::transform(word.begin(), word.end(), word.begin(), ::tolower);

    return true;
}

// Clean text content (remove YAML frontmatter and HTML tags)
std::string CleanText(const std::string& rawText)
{
    std::string text = rawText;

    // 1. Remove YAML Frontmatter (lines between --- at the start)
    // Regex: Start of string, ---, anything non-greedy, ---
    static const std::regex frontmatterRegex(R"(^---\s*[\r\n]+[\s\S]*?[\r\n]+---\s*)");
    text = std::regex_replace(text, frontmatterRegex, "");

    // 2. Remove HTML tags
    // Regex: < followed by any character that is not >, followed by >
    static const std::regex htmlTagRegex(R"(<[^>]*>)");
    text = std::regex_replace(text, htmlTagRegex, " ");

    return text;
}

template <size_t ORDER_N>
bool ProcessFile(const std::string& fileName, MarkovChain<std::string, ORDER_N>& markovChain)
{
    // read the file into memory
    FILE* file = fopen(fileName.c_str(), "rt");
    if (!file)
        return false;
    
    std::vector<char> rawContents;
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    if (fileSize > 0)
    {
        rawContents.resize(fileSize);
        fseek(file, 0, SEEK_SET);
        fread(rawContents.data(), 1, fileSize, file);
    }
    fclose(file);

    if (rawContents.empty()) return true;

    // Convert to string for regex processing
    std::string contentStr(rawContents.begin(), rawContents.end());

    // Clean the text
    std::string cleanedContent = CleanText(contentStr);

    // Copy back to a buffer we can iterate over easily for GetWord
    // (GetWord expects unsigned char*, strictly speaking, but char* is fine for cast)
    std::vector<unsigned char> contents(cleanedContent.begin(), cleanedContent.end());

    // get an observation context
    auto context = markovChain.GetObservationContext();

    // process the file
    size_t position = 0;
    size_t size = contents.size();
    std::string nextWord;
    while(GetWord(contents.data(), size, position, nextWord))
        markovChain.RecordObservation(context, nextWord);

    return true;
}

template <typename MARKOVCHAIN>
bool GenerateStatsFileTemplated(const char* fileName, MARKOVCHAIN& markovChain)
{
    FILE* file = fopen(fileName, "w+t");
    if (!file)
        return false;

    // show the data we have
    fprintf(file, "\n\nWord Counts:");
    for (auto& wordCounts : markovChain.m_counts)
    {
        // fprintf for Observations is defined earlier
        fprintf(file, "\n[+] ");
        fprintf(file, wordCounts.first);
        fprintf(file, "\n");

        for (auto& wordCount : wordCounts.second)
            fprintf(file, "[++] %s - %zu\n", wordCount.first.c_str(), wordCount.second);
    }

    fprintf(file, "\n\nWord Probabilities:");
    for (auto& wordCounts : markovChain.m_probabilities)
    {
        fprintf(file, "\n[-] ");
        fprintf(file, wordCounts.first);
        fprintf(file, "\n");

        float lastProbability = 0.0f;
        for (auto& wordCount : wordCounts.second)
        {
            fprintf(file, "[--] %s - %i%%\n", wordCount.observed.c_str(), int((wordCount.cumulativeProbability - lastProbability)*100.0f));
            lastProbability = wordCount.cumulativeProbability;
        }
    }
    
    fclose(file);
    return true;
}

template <typename MARKOVCHAIN>
bool GenerateFile(const char* fileName, size_t wordCount, MARKOVCHAIN& markovChain)
{
    FILE* file = fopen(fileName, "w+t");
    if (!file)
        return false;

    // get the initial starting state
    auto observations = markovChain.GetInitialObservations();
    std::string word = observations[0];
    if (!word.empty()) word[0] = toupper(word[0]);
    fprintf(file, "%s", word.c_str());

    bool capitalizeFirstLetter = false;

    for (size_t wordIndex = 0; wordIndex < wordCount; ++wordIndex)
    {
        markovChain.GetNextObservations(observations);

        std::string nextWord = observations[0];
        if (capitalizeFirstLetter && !nextWord.empty())
        {
            nextWord[0] = toupper(nextWord[0]);
            capitalizeFirstLetter = false;
        }

        if (nextWord == "." || nextWord == "," || nextWord == ";" || nextWord == ":")
            fprintf(file, "%s", nextWord.c_str());
        else
            fprintf(file, " %s", nextWord.c_str());

        if (nextWord == ".")
            capitalizeFirstLetter = true;
    }

    fclose(file);
    return true;
}

// Specialization or overload for RunEngine to call the templated stats
template <size_t ORDER_N>
int RunEngineWithStats(const std::vector<std::string>& inputFiles, int wordCount)
{
    MarkovChain<std::string, ORDER_N> markovChain;

    for (const auto& inputFile : inputFiles)
    {
        printf("processing %s...\n", inputFile.c_str());
        if (!ProcessFile(inputFile, markovChain))
        {
            printf("could not open file %s!\n", inputFile.c_str());
        }
    }

    printf("Calculating probabilities (Order %zu)...\n", ORDER_N);
    markovChain.FinalizeLearning();

    if (!GenerateStatsFileTemplated("out/stats.txt", markovChain))
    {
        printf("Could not generate stats file!\n");
    }

    if (!GenerateFile("out/generated.txt", wordCount, markovChain))
    {
        printf("Could not generate output file!\n");
        return 1;
    }

    printf("Done. Output in out/generated.txt\n");
    return 0;
}

void PrintUsage()
{
    printf("Usage: textgen [-o order] [-l length]\n");
    printf("  -o order   : Markov chain order (1-5). Default 2.\n");
    printf("  -l length  : Number of words to generate. Default 1000.\n");
}

int main(int argc, char** argv)
{
    // Default parameters
    int order = 2;
    int wordCount = 1000;

    // Simple arg parsing
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc)
        {
            order = std::atoi(argv[++i]);
        }
        else if (arg == "-l" && i + 1 < argc)
        {
            wordCount = std::atoi(argv[++i]);
        }
        else
        {
            PrintUsage();
            return 1;
        }
    }

    // Find files
    std::vector<std::string> inputFiles;
    if (std::filesystem::exists("data"))
    {
        for (const auto& entry : std::filesystem::directory_iterator("data"))
        {
            std::string ext = entry.path().extension().string();
            if (ext == ".txt" || ext == ".md")
            {
                inputFiles.push_back(entry.path().string());
            }
        }
    }
    
    if (inputFiles.empty())
    {
        printf("No .txt or .md files found in data/ directory.\n");
        return 1;
    }

    // Run appropriate engine
    switch (order)
    {
    case 1: return RunEngineWithStats<1>(inputFiles, wordCount);
    case 2: return RunEngineWithStats<2>(inputFiles, wordCount);
    case 3: return RunEngineWithStats<3>(inputFiles, wordCount);
    case 4: return RunEngineWithStats<4>(inputFiles, wordCount);
    case 5: return RunEngineWithStats<5>(inputFiles, wordCount);
    default:
        printf("Invalid order %d. Supported orders: 1-5.\n", order);
        return 1;
    }
}

/*

Next: try with images?
 * maybe just have a "N observed states = M possible outputs" general markov chain. Maybe try it with more than images? letters? audio? i dunno

Note: all sorts of copies and ineficiencies in code. Runs fast enough for this usage case, and was fast to write, so good enough.
Note: try 0th order (purely random selection of words), 2nd order, 3rd order, etc. Show how it limits options. Need more data i guess.
Note: hitting situations where there is nothing to transition to next? actually i dont think this is an issue

*/