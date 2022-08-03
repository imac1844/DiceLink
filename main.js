Hooks.on('init', () => {
    game.settings.register("manual-rolls", "manualRolls", {
        name: "Roll Dice Manually",
        hint: "Enable this to roll dice offline and report the result.",
        scope: "client",
        config: true,
        default: false,
        type: Boolean,
    });
});

Hooks.once('ready', () => {
	const baseRoll = Die.prototype.roll;
	Die.prototype.roll = function(minimize=false, maximize=false) {
		if (game.settings.get('manual-rolls','manualRolls')) {
			const promptText = this.number > 1 ? 'Roll '+this.number+'d'+this.faces+' #'+(this.results.length + 1) : 'Roll a d'+this.faces;
			this.results.push({result: Number(prompt(promptText)), active: true});
			return this;
		}
		else {
			const roll = baseRoll.bind(this);
			roll(minimize, maximize);
		}
}});