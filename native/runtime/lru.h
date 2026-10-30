#pragma once
#include <cstddef>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

// LRU cache of shared_ptr values keyed by string. Not synchronized.
template <typename V>
class LruCache {
   public:
    explicit LruCache(size_t limit) : limit_(limit) {}

    std::shared_ptr<V> find(const std::string& key) {
        auto it = entries_.find(key);
        if (it == entries_.end()) return nullptr;
        order_.splice(order_.begin(), order_, it->second.second);
        return it->second.first;
    }

    // Returns the existing entry if the key is already present.
    std::shared_ptr<V> insert(const std::string& key, std::shared_ptr<V> value) {
        if (auto existing = find(key)) return existing;
        order_.push_front(key);
        entries_.emplace(key, std::make_pair(value, order_.begin()));
        evict();
        return value;
    }

    // 0 disables eviction.
    void set_limit(size_t limit) {
        limit_ = limit;
        evict();
    }
    size_t size() const { return entries_.size(); }
    void clear() {
        entries_.clear();
        order_.clear();
    }

   private:
    void evict() {
        if (limit_ == 0) return;
        while (entries_.size() > limit_) {
            entries_.erase(order_.back());
            order_.pop_back();
        }
    }

    using Order = std::list<std::string>;  // front = most recently used
    size_t limit_;
    Order order_;
    std::unordered_map<std::string, std::pair<std::shared_ptr<V>, Order::iterator>> entries_;
};
