export class ProductController {
    #productView;
    #currentUser = null;
    #events;
    #productService;
    constructor({
        productView,
        events,
        productService
    }) {
        this.#productView = productView;
        this.#productService = productService;
        this.#events = events;
        this.init();
    }

    static init(deps) {
        return new ProductController(deps);
    }

    async init() {
        this.setupCallbacks();
        this.setupEventListeners();
        const products = await this.#productService.getProducts();
        this.#productView.render(products, true);
    }

    setupEventListeners() {

        this.#events.onUserSelected((user) => {
            this.#currentUser = user;
            this.#productView.onUserSelected(user);
            this.#events.dispatchRecommend(user)
        })

        this.#events.onRecommendationsReady(({ recommendations }) => {
            // Filtrar recomendações: remover produtos que já estão em Past Purchases
            const purchasedProductIds = new Set(
                this.#currentUser.purchases.map(p => p.id)
            );
            const filteredRecommendations = recommendations.filter(
                rec => !purchasedProductIds.has(rec.id)
            );
            this.#productView.render(filteredRecommendations, false);
        });
    }

    setupCallbacks() {
        this.#productView.registerBuyProductCallback(this.handleBuyProduct.bind(this));
    }

    async handleBuyProduct(product) {
        const user = this.#currentUser;
        this.#events.dispatchPurchaseAdded({ user, product });
    }

}
