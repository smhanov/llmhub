package llmhub

import (
	"fmt"
	"sort"
	"sync"
)

// ProviderFactory describes a function that can instantiate a provider backed by a specific vendor SDK.
type ProviderFactory func(apiKey string, opts ...Option) (Provider, error)

var (
	registryMu       sync.RWMutex
	providerRegistry = map[string]ProviderFactory{}
)

// RegisterProvider adds a provider factory to the global registry.
func RegisterProvider(name string, factory ProviderFactory) error {
	if name == "" {
		return fmt.Errorf("%w: name is empty", ErrInvalidInput)
	}
	if factory == nil {
		return fmt.Errorf("%w: factory is nil", ErrInvalidInput)
	}
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, exists := providerRegistry[name]; exists {
		return fmt.Errorf("%w: %s", ErrProviderAlreadyRegistered, name)
	}
	providerRegistry[name] = factory
	return nil
}

// MustRegisterProvider registers a provider and panics on failure.
func MustRegisterProvider(name string, factory ProviderFactory) {
	if err := RegisterProvider(name, factory); err != nil {
		panic(err)
	}
}

// RegisteredProviders returns a sorted slice of registered provider names.
func RegisteredProviders() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	names := make([]string, 0, len(providerRegistry))
	for name := range providerRegistry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// lookupProvider fetches the factory for a provider name.
func lookupProvider(name string) (ProviderFactory, error) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	factory, ok := providerRegistry[name]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrProviderNotFound, name)
	}
	return factory, nil
}

// New creates a Client bound to the named provider.
func New(providerName, apiKey string, opts ...Option) (*Client, error) {
	factory, err := lookupProvider(providerName)
	if err != nil {
		return nil, err
	}
	provider, err := factory(apiKey, opts...)
	if err != nil {
		return nil, err
	}
	return &Client{provider: provider, defaultOpts: append([]Option(nil), opts...)}, nil
}

// MustNew is a helper that panics if client creation fails.
func MustNew(providerName, apiKey string, opts ...Option) *Client {
	client, err := New(providerName, apiKey, opts...)
	if err != nil {
		panic(err)
	}
	return client
}
